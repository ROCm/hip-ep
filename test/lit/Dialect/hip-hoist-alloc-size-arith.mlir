// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-hoist-alloc-size-arith.
//
// The pass moves speculatable producers of `memref.alloc` dynamic operands
// above the earliest dynamic `memref.alloc` in the function's single
// entry block.  After the pass, every dyn-operand SSA def dominates the
// earliest dynamic alloc — which is the precondition `--hip-pool-allocs`'s
// single-block dominator-emit phase needs.
//
// These tests cover the pass in isolation (no PoolAllocs).  PoolAllocs's
// own LIT remains unchanged — it consumes already-hoisted IR.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-hoist-alloc-size-arith %s 2>&1 | FileCheck %s

// --- 1. Pure arith.muli between two dynamic allocs.  Both `%6` and
//        `%7` are speculatable, both are below `%alloc`.  After the
//        pass they appear above `%alloc` in operand-before-use order.
// CHECK-LABEL: func.func @hoist_simple_chain
// CHECK:         %[[DIM0:.*]] = memref.dim
// CHECK:         %[[DIM1:.*]] = memref.dim
// CHECK:         %[[MUL0:.*]] = arith.muli %[[DIM0]], %[[DIM1]]
// CHECK:         %[[MUL1:.*]] = arith.muli %[[MUL0]], %{{.*}}
// CHECK:         %[[ALLOC0:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]])
// CHECK:         %[[ALLOC1:.*]] = memref.alloc(%[[MUL1]])
// CHECK:         memref.dealloc %[[ALLOC0]]
// CHECK:         memref.dealloc %[[ALLOC1]]
func.func @hoist_simple_chain(%arg0: memref<?x?xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4096 = arith.constant 4096 : index
  %dim = memref.dim %arg0, %c0 : memref<?x?xi64>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?xi64>
  %alloc = memref.alloc(%dim, %dim_0) : memref<?x?xui8>
  %6 = arith.muli %dim, %dim_0 : index
  %7 = arith.muli %6, %c4096 : index
  %alloc_6 = memref.alloc(%7) : memref<3x?xi64>
  memref.dealloc %alloc : memref<?x?xui8>
  memref.dealloc %alloc_6 : memref<3x?xi64>
  return
}

// --- 2. Mixed-arith chain (addi / subi / divui / index_cast / select).
//        Every op is speculatable when its operands are; the pass
//        hoists the entire chain above the earliest dynamic alloc and
//        preserves operand-before-use order.  `arith.divui` is
//        `ConditionallySpeculatable` — `mlir::isSpeculatable` returns
//        true here because the divisor is a known-non-zero constant.
// CHECK-LABEL: func.func @hoist_mixed_arith_chain
// CHECK:         %[[DIM:.*]] = memref.dim
// CHECK:         %[[ADD:.*]] = arith.addi %[[DIM]], %{{.*}}
// CHECK:         %[[SUB:.*]] = arith.subi %[[ADD]], %{{.*}}
// CHECK:         %[[DIV:.*]] = arith.divui %[[SUB]], %{{.*}}
// CHECK:         %[[CMP:.*]] = arith.cmpi sgt, %[[DIV]], %{{.*}}
// CHECK:         %[[SEL:.*]] = arith.select %[[CMP]], %[[DIV]], %{{.*}}
// CHECK:         %[[ALLOC0:.*]] = memref.alloc(%[[DIM]])
// CHECK:         %[[ALLOC1:.*]] = memref.alloc(%[[SEL]])
func.func @hoist_mixed_arith_chain(%arg0: memref<?xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %dim = memref.dim %arg0, %c0 : memref<?xi64>
  %alloc = memref.alloc(%dim) : memref<?xf16>
  %add = arith.addi %dim, %c8 : index
  %sub = arith.subi %add, %c1 : index
  %div = arith.divui %sub, %c2 : index
  %cmp = arith.cmpi sgt, %div, %c1 : index
  %sel = arith.select %cmp, %div, %c1 : index
  %alloc_1 = memref.alloc(%sel) : memref<?xf16>
  memref.dealloc %alloc : memref<?xf16>
  memref.dealloc %alloc_1 : memref<?xf16>
  return
}

// --- 3. `memref.load` between two dynamic allocs.  `memref.load` is
//        NOT speculatable (read effect), so the pass must NOT hoist it.
//        Verifies the side-effect filter.
// CHECK-LABEL: func.func @do_not_hoist_load
// CHECK:         %[[DIM:.*]] = memref.dim
// CHECK-NEXT:    %[[ALLOC0:.*]] = memref.alloc(%[[DIM]])
// CHECK-NEXT:    %[[LOAD:.*]] = memref.load
// CHECK-NEXT:    %[[CAST:.*]] = arith.index_cast %[[LOAD]]
// CHECK-NEXT:    %[[ALLOC1:.*]] = memref.alloc(%[[CAST]])
func.func @do_not_hoist_load(%arg0: memref<?xi64>, %src: memref<1xi64>) {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?xi64>
  %alloc = memref.alloc(%dim) : memref<?xf16>
  %loaded = memref.load %src[%c0] : memref<1xi64>
  %cast = arith.index_cast %loaded : i64 to index
  %alloc_1 = memref.alloc(%cast) : memref<?xf16>
  memref.dealloc %alloc : memref<?xf16>
  memref.dealloc %alloc_1 : memref<?xf16>
  return
}

// --- 4. `func.call` with no Pure trait between two dynamic allocs.
//        Without `Pure` / `MemoryEffects::None` on the callee, the call
//        is conservatively non-speculatable and must NOT be hoisted.
//        Verifies the callable side-effect filter.
// CHECK-LABEL: func.func @do_not_hoist_unmarked_call
// CHECK:         %[[DIM:.*]] = memref.dim
// CHECK-NEXT:    %[[ALLOC0:.*]] = memref.alloc(%[[DIM]])
// CHECK-NEXT:    %[[CALL:.*]] = call @opaque_helper
// CHECK-NEXT:    %[[ALLOC1:.*]] = memref.alloc(%[[CALL]])
func.func private @opaque_helper(%x: index) -> index

func.func @do_not_hoist_unmarked_call(%arg0: memref<?xi64>) {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?xi64>
  %alloc = memref.alloc(%dim) : memref<?xf16>
  %r = func.call @opaque_helper(%dim) : (index) -> index
  %alloc_1 = memref.alloc(%r) : memref<?xf16>
  memref.dealloc %alloc : memref<?xf16>
  memref.dealloc %alloc_1 : memref<?xf16>
  return
}

// --- 5. Already-hoisted IR.  Every dyn-operand def is above the
//        earliest dynamic alloc — the pass is a no-op.  Verifies
//        idempotence.
// CHECK-LABEL: func.func @already_hoisted
// CHECK:         %[[DIM:.*]] = memref.dim
// CHECK-NEXT:    %[[MUL:.*]] = arith.muli %[[DIM]], %{{.*}}
// CHECK-NEXT:    %[[ALLOC0:.*]] = memref.alloc(%[[DIM]])
// CHECK-NEXT:    %[[ALLOC1:.*]] = memref.alloc(%[[MUL]])
func.func @already_hoisted(%arg0: memref<?xi64>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %dim = memref.dim %arg0, %c0 : memref<?xi64>
  %mul = arith.muli %dim, %c2 : index
  %alloc = memref.alloc(%dim) : memref<?xf16>
  %alloc_1 = memref.alloc(%mul) : memref<?xf16>
  memref.dealloc %alloc : memref<?xf16>
  memref.dealloc %alloc_1 : memref<?xf16>
  return
}

// --- 6. Mixed chain: `arith.muli %loaded, %c2` is itself speculatable
//        but its operand `%loaded` (memref.load) is NOT.  The pass
//        must NOT hoist `%mul` either — the recursive operand check
//        bails the whole chain out.  Verifies "stop-at-non-speculatable"
//        propagates upward through the chain.
// CHECK-LABEL: func.func @do_not_hoist_through_load
// CHECK:         %[[DIM:.*]] = memref.dim
// CHECK-NEXT:    %[[ALLOC0:.*]] = memref.alloc(%[[DIM]])
// CHECK-NEXT:    %[[LOAD:.*]] = memref.load
// CHECK-NEXT:    %[[CAST:.*]] = arith.index_cast %[[LOAD]]
// CHECK-NEXT:    %[[MUL:.*]] = arith.muli %[[CAST]], %{{.*}}
// CHECK-NEXT:    %[[ALLOC1:.*]] = memref.alloc(%[[MUL]])
func.func @do_not_hoist_through_load(%arg0: memref<?xi64>,
                                     %src: memref<1xi64>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %dim = memref.dim %arg0, %c0 : memref<?xi64>
  %alloc = memref.alloc(%dim) : memref<?xf16>
  %loaded = memref.load %src[%c0] : memref<1xi64>
  %cast = arith.index_cast %loaded : i64 to index
  %mul = arith.muli %cast, %c2 : index
  %alloc_1 = memref.alloc(%mul) : memref<?xf16>
  memref.dealloc %alloc : memref<?xf16>
  memref.dealloc %alloc_1 : memref<?xf16>
  return
}
