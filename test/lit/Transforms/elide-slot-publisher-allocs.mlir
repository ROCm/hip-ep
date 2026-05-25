// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// elide-slot-publisher-allocs.mlir
//
// Phase 1 of the slot-buffer-coalescing initiative
// (docs/design/slot-buffer-coalesce.md). The pass shrinks the dynamic-
// size operands of memref.alloc ops that back the DPS-init of a Cat-C
// slot publisher (NonZero, Range Cat-C, ConstantOfShape Cat-C). The
// alloc's MemRefType is preserved; only its runtime extent is
// collapsed to zero so the subsequent --hip-pool-allocs treats it as
// a 0-byte dynamic bucket and the static pool footprint shrinks.
//
// Two complementary test groups:
//   * SHRINK: dynamic-size operand of a hand-written `hip.nonzero` with
//     `hipdnn.elide_dps_init` is rewritten from a runtime upper-bound
//     SSA value (`%ub`) to `arith.constant 0`. No alloc is erased --
//     we keep the SSA edge intact for the DPS contract.
//   * SKIP:   ops WITHOUT the marker (or whose alloc has at least one
//     non-hip, non-memref, non-dim, non-dealloc user) are left alone.

// RUN: hip-mlir-opt --split-input-file --hip-elide-slot-publisher-allocs %s | FileCheck %s

// === SHRINK ====
//
// Hand-rolled mini-IR mirroring what bufferize-then-canonicalize produces
// for a Cat-C NonZero with an upper bound %ub. The pass should rewrite
// every dynamic-size operand of `%alloc` to `arith.constant 0`.
//
// CHECK-LABEL: func.func @shrink_nonzero
// CHECK-SAME:   (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: memref<3x4xi1>, %[[UB:.*]]: index)
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[ALLOC:.*]] = memref.alloc(%[[C0]]) : memref<2x?xi64>
// CHECK:        hip.nonzero(%[[CTX]]) ins(%[[IN]] : memref<3x4xi1>) outs(%[[ALLOC]] : memref<2x?xi64>)
// CHECK-SAME:   hipdnn.elide_dps_init
// CHECK:        return
func.func @shrink_nonzero(%ctx: !hip.context, %input: memref<3x4xi1>, %ub: index) {
  %alloc = memref.alloc(%ub) : memref<2x?xi64>
  hip.nonzero(%ctx) ins(%input : memref<3x4xi1>) outs(%alloc : memref<2x?xi64>) {
    hipdnn.elide_dps_init,
    input_data_type = 5 : i64,
    slot_id = 0 : i32
  }
  memref.dealloc %alloc : memref<2x?xi64>
  return
}

// -----

// === SHRINK with downstream hip consumer (the realistic post-bufferize layout) ===
//
// Mirrors NonZero -> Transpose chain. The Transpose reads %alloc; the
// pass must still shrink because the Transpose is a hip-dialect op
// that AnnotateInputDimSlotsPass (later in the pipeline) will wire to
// the slot publisher.
//
// CHECK-LABEL: func.func @shrink_with_hip_consumer
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[ALLOC:.*]] = memref.alloc(%[[C0]]) : memref<2x?xi64>
// CHECK:        hip.nonzero
// CHECK-SAME:   hipdnn.elide_dps_init
// CHECK:        hip.transpose
func.func @shrink_with_hip_consumer(
    %ctx: !hip.context, %input: memref<3x4xi1>, %ub: index,
    %tout: memref<?x2xi64>) {
  %alloc = memref.alloc(%ub) : memref<2x?xi64>
  hip.nonzero(%ctx) ins(%input : memref<3x4xi1>) outs(%alloc : memref<2x?xi64>) {
    hipdnn.elide_dps_init,
    input_data_type = 5 : i64,
    slot_id = 1 : i32
  }
  hip.transpose(%ctx) ins(%alloc : memref<2x?xi64>) outs(%tout : memref<?x2xi64>) {
    perm = [1, 0]
  }
  memref.dealloc %alloc : memref<2x?xi64>
  return
}

// -----

// === SKIP: no marker ===
//
// The same hip.nonzero shape but WITHOUT `hipdnn.elide_dps_init`. The
// pass must not touch the alloc -- its upper-bound %ub must survive
// unchanged.
//
// CHECK-LABEL: func.func @skip_no_marker
// CHECK-SAME:   (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: memref<3x4xi1>, %[[UB:.*]]: index)
// CHECK:        %[[ALLOC:.*]] = memref.alloc(%[[UB]]) : memref<2x?xi64>
// CHECK-NOT:    arith.constant 0 : index
// CHECK:        hip.nonzero
func.func @skip_no_marker(%ctx: !hip.context, %input: memref<3x4xi1>, %ub: index) {
  %alloc = memref.alloc(%ub) : memref<2x?xi64>
  hip.nonzero(%ctx) ins(%input : memref<3x4xi1>) outs(%alloc : memref<2x?xi64>) {
    input_data_type = 5 : i64,
    slot_id = 2 : i32
  }
  memref.dealloc %alloc : memref<2x?xi64>
  return
}

// -----

// === SKIP: opaque external user ===
//
// Marker present, but `func.return` carries the alloc out of the
// function. That's outside the hip / memref dialect and the safety
// walk treats it as opaque -- the pass must skip. The alloc's UB
// stays intact.
//
// CHECK-LABEL: func.func @skip_opaque_user
// CHECK-SAME:   (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: memref<3x4xi1>, %[[UB:.*]]: index)
// CHECK:        %[[ALLOC:.*]] = memref.alloc(%[[UB]]) : memref<2x?xi64>
// CHECK-NOT:    arith.constant 0 : index
// CHECK:        hip.nonzero
// CHECK:        return %[[ALLOC]]
func.func @skip_opaque_user(%ctx: !hip.context, %input: memref<3x4xi1>, %ub: index)
    -> memref<2x?xi64> {
  %alloc = memref.alloc(%ub) : memref<2x?xi64>
  hip.nonzero(%ctx) ins(%input : memref<3x4xi1>) outs(%alloc : memref<2x?xi64>) {
    hipdnn.elide_dps_init,
    input_data_type = 5 : i64,
    slot_id = 3 : i32
  }
  return %alloc : memref<2x?xi64>
}
