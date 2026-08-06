// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-hoist-alloc-size-arith.
//
// The pass moves pure producers of `memref.alloc` dynamic operands
// above the earliest used `memref.alloc` in the function's single entry block.
// After the pass, every hoistable dynamic-size value dominates every
// allocation that `--hip-pool-allocs` may absorb.
//
// These tests cover the pass in isolation (no PoolAllocs).  PoolAllocs's
// own LIT remains unchanged — it consumes already-hoisted IR.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-hoist-alloc-size-arith %s 2>&1 | FileCheck %s

// --- 1. Pure arith.muli between two dynamic allocs. Both `%6` and
//        `%7` are pure, both are below `%alloc`. After the
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
//        Every op is pure when its operands are; the pass
//        hoists the entire chain above the earliest used alloc and
//        preserves operand-before-use order.  `arith.divui` is
//        `ConditionallySpeculatable`; `mlir::isPure` returns true here because
//        the divisor is a known-non-zero constant and the op has no effects.
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

// --- 3. `memref.load` between two dynamic allocs is not pure because it has a
//        read effect, so the pass must not hoist it.
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
//        is conservatively not pure and must not be hoisted.
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

// --- 5. Already-hoisted IR. Every dynamic-size definition is above the
//        earliest used allocation, so the pass is a no-op.
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

// --- 6. `arith.muli %loaded, %c2` is pure, but its `memref.load` operand is
//        not. The recursive check therefore keeps the complete chain in place.
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

// --- 7. `arith.divsi` with a *runtime* (non-constant) divisor between two
//        dynamic allocs.  `arith.divsi` is `ConditionallySpeculatable`:
//        its interface cannot prove this runtime divisor safe to speculate.
//        `mlir::isPure` therefore returns false and the pass must not hoist
//        `%div`.
//        The two pure `memref.dim` operands already dominate the alloc, so
//        they are untouched.  This is the negative counterpart to case 2
//        and guards the "div with runtime divisor stays put" invariant
//        called out in the pass's header comment.
// CHECK-LABEL: func.func @do_not_hoist_runtime_divisor_divsi
// CHECK:         %[[DIM:.*]] = memref.dim
// CHECK-NEXT:    %[[DIVISOR:.*]] = memref.dim
// CHECK-NEXT:    %[[ALLOC0:.*]] = memref.alloc(%[[DIM]])
// CHECK-NEXT:    %[[DIV:.*]] = arith.divsi %[[DIM]], %[[DIVISOR]]
// CHECK-NEXT:    %[[ALLOC1:.*]] = memref.alloc(%[[DIV]])
func.func @do_not_hoist_runtime_divisor_divsi(%arg0: memref<?xi64>,
                                              %arg1: memref<?xi64>) {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?xi64>
  %divisor = memref.dim %arg1, %c0 : memref<?xi64>
  %alloc = memref.alloc(%dim) : memref<?xf16>
  %div = arith.divsi %dim, %divisor : index
  %alloc_1 = memref.alloc(%div) : memref<?xf16>
  memref.dealloc %alloc : memref<?xf16>
  memref.dealloc %alloc_1 : memref<?xf16>
  return
}

// --- 8. Same as case 7 for `arith.divui`.  Both signed and unsigned
//        integer division are `ConditionallySpeculatable` and both trap on
//        a zero divisor, so a runtime divisor blocks the hoist identically.
// CHECK-LABEL: func.func @do_not_hoist_runtime_divisor_divui
// CHECK:         %[[DIM:.*]] = memref.dim
// CHECK-NEXT:    %[[DIVISOR:.*]] = memref.dim
// CHECK-NEXT:    %[[ALLOC0:.*]] = memref.alloc(%[[DIM]])
// CHECK-NEXT:    %[[DIV:.*]] = arith.divui %[[DIM]], %[[DIVISOR]]
// CHECK-NEXT:    %[[ALLOC1:.*]] = memref.alloc(%[[DIV]])
func.func @do_not_hoist_runtime_divisor_divui(%arg0: memref<?xi64>,
                                              %arg1: memref<?xi64>) {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?xi64>
  %divisor = memref.dim %arg1, %c0 : memref<?xi64>
  %alloc = memref.alloc(%dim) : memref<?xf16>
  %div = arith.divui %dim, %divisor : index
  %alloc_1 = memref.alloc(%div) : memref<?xf16>
  memref.dealloc %alloc : memref<?xf16>
  memref.dealloc %alloc_1 : memref<?xf16>
  return
}

// --- 9. A static allocation precedes a block-argument dim query used by a
//        later dynamic allocation. PoolAllocs absorbs both allocations, so the
//        dim must move above the static allocation.
// CHECK-LABEL: func.func @hoist_above_earliest_static_alloc
// CHECK:         %[[DIM:.*]] = memref.dim
// CHECK-NEXT:    %[[STATIC:.*]] = memref.alloc()
// CHECK-NEXT:    %[[DYNAMIC:.*]] = memref.alloc(%[[DIM]])
// CHECK:         memref.dealloc %[[STATIC]]
// CHECK:         memref.dealloc %[[DYNAMIC]]
func.func @hoist_above_earliest_static_alloc(%arg0: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %static = memref.alloc() : memref<1xi32>
  %dim = memref.dim %arg0, %c0 : memref<?xf32>
  %dynamic = memref.alloc(%dim) : memref<?xf32>
  memref.dealloc %static : memref<1xi32>
  memref.dealloc %dynamic : memref<?xf32>
  return
}

// --- 10. A recursively-speculatable region operation implicitly captures
//         %late. The capture is not a parent-op operand, so moving the scf.if
//         above %late would violate dominance inside its region. Region-bearing
//         producers are therefore never hoisted.
// CHECK-LABEL: func.func @do_not_hoist_region_capture
// CHECK:         %[[STATIC:.*]] = memref.alloc()
// CHECK-NEXT:    %[[DIM:.*]] = memref.dim
// CHECK-NEXT:    %[[LATE:.*]] = arith.addi
// CHECK-NEXT:    %[[SIZE:.*]] = scf.if
// CHECK:         %[[DYNAMIC:.*]] = memref.alloc(%[[SIZE]])
func.func @do_not_hoist_region_capture(
    %arg0: memref<?xf32>, %condition: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %static = memref.alloc() : memref<1xi32>
  %dim = memref.dim %arg0, %c0 : memref<?xf32>
  %late = arith.addi %dim, %c1 : index
  %size = scf.if %condition -> (index) {
    scf.yield %late : index
  } else {
    scf.yield %dim : index
  }
  %dynamic = memref.alloc(%size) : memref<?xf32>
  memref.dealloc %static : memref<1xi32>
  memref.dealloc %dynamic : memref<?xf32>
  return
}

// --- 11. One branch of the size cone is pure and the other depends on a
//         load. Rejecting the full cone must not move the pure branch alone.
// CHECK-LABEL: func.func @do_not_hoist_partial_cone
// CHECK:         %[[STATIC:.*]] = memref.alloc()
// CHECK-NEXT:    %[[DIM:.*]] = memref.dim
// CHECK-NEXT:    %[[PURE:.*]] = arith.addi
// CHECK-NEXT:    %[[LOAD:.*]] = memref.load
// CHECK-NEXT:    %[[CAST:.*]] = arith.index_cast %[[LOAD]]
// CHECK-NEXT:    %[[SIZE:.*]] = arith.addi %[[PURE]], %[[CAST]]
// CHECK-NEXT:    %[[DYNAMIC:.*]] = memref.alloc(%[[SIZE]])
func.func @do_not_hoist_partial_cone(
    %arg0: memref<?xf32>, %shape: memref<1xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %static = memref.alloc() : memref<1xi32>
  %dim = memref.dim %arg0, %c0 : memref<?xf32>
  %pure = arith.addi %dim, %c1 : index
  %loaded = memref.load %shape[%c0] : memref<1xi64>
  %cast = arith.index_cast %loaded : i64 to index
  %size = arith.addi %pure, %cast : index
  %dynamic = memref.alloc(%size) : memref<?xf32>
  memref.dealloc %static : memref<1xi32>
  memref.dealloc %dynamic : memref<?xf32>
  return
}
