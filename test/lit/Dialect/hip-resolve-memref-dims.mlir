// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck test for --hip-resolve-memref-dims and its effect on the
// --hip-pool-allocs dominance-domain count.
//
// The barrier: an alloc whose dynamic size is `memref.dim` of a mid-block
// `memref.subview`.  The subview's dim 0 equals the function argument's dim 0
// (%d0, computed at block top), but written as a dim of a view defined BELOW
// the earliest pooled alloc, so --hip-hoist-alloc-size-arith cannot lift it
// (the cone bottoms out at %a0) and --hip-pool-allocs splits a second domain.
//
// --hip-resolve-memref-dims folds `memref.dim %sv, 0` -> `memref.dim %arg, 0`
// (== %d0), so the second alloc is sized by a block-top value and both allocs
// share ONE dominance domain / ONE hip.get_pool.
//===----------------------------------------------------------------------===//

// Baseline (no resolution): the dim-of-view barrier forces two domains.
// RUN: hip-mlir-opt --hip-pool-allocs %s | FileCheck %s --check-prefix=BASE

// The fold itself: `memref.dim` of the subview is rewritten to a dim of %arg.
// RUN: hip-mlir-opt --hip-resolve-memref-dims %s | FileCheck %s --check-prefix=FOLD

// End to end: resolve + hoist + pool collapses to a single domain / pool.
// RUN: hip-mlir-opt --hip-resolve-memref-dims --hip-hoist-alloc-size-arith \
// RUN:   --hip-pool-allocs %s | FileCheck %s --check-prefix=FIXED

// BASE: hipdnn.domain_count = 2
// BASE-LABEL: func.func @dim_of_view
// BASE-COUNT-2: hip.get_pool

// FOLD-LABEL: func.func @dim_of_view
// The subview stays (it still feeds the copy), but the `memref.dim` query on
// the strided view is folded away to a dim of the function argument.
// FOLD: memref.subview
// FOLD-NOT: memref.dim{{.*}}strided

// FIXED-LABEL: func.func @dim_of_view
// FIXED-NOT: hipdnn.domain_count
// FIXED-COUNT-1: hip.get_pool
// FIXED-NOT: hip.get_pool

func.func @dim_of_view(
    %ctx: !hip.context,
    %arg1: memref<?x8192xf16>,
    %dst: memref<?x4096xf16>) -> memref<?x4096xf16> {
  %c0 = arith.constant 0 : index

  // Block-top dim of the function argument.
  %d0 = memref.dim %arg1, %c0 : memref<?x8192xf16>

  // Earliest pooled alloc.
  %a0 = memref.alloc(%d0) : memref<?x8192xf16>
  memref.copy %arg1, %a0 : memref<?x8192xf16> to memref<?x8192xf16>

  // Mid-block strided view (the "V slice"): defined BELOW %a0.
  %sv = memref.subview %a0[0, 4096] [%d0, 4096] [1, 1]
      : memref<?x8192xf16> to memref<?x4096xf16, strided<[8192, 1], offset: 4096>>

  // Size of the second alloc comes from `dim` of the mid-block view -> barrier.
  %vd0 = memref.dim %sv, %c0
      : memref<?x4096xf16, strided<[8192, 1], offset: 4096>>
  %a1 = memref.alloc(%vd0) : memref<?x4096xf16>
  memref.copy %sv, %a1
      : memref<?x4096xf16, strided<[8192, 1], offset: 4096>> to memref<?x4096xf16>

  memref.copy %a1, %dst : memref<?x4096xf16> to memref<?x4096xf16>
  return %dst : memref<?x4096xf16>
}
