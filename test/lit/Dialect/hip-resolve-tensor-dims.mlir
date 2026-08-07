// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt %s --hip-resolve-tensor-dims | FileCheck %s
//
// Pins down the `tensor.dim` folds `--hip-resolve-tensor-dims` is
// expected to handle.  The pass is a thin wrapper over upstream MLIR's
// reify-driven patterns plus tensor canonicalisers, and depends on
// `tensor::registerInferTypeOpInterfaceExternalModels` being wired
// into the dialect registry (see InitAllPasses.h /
// hip-mlir-opt.cpp).  Coverage:
//
//   * `tensor.dim` of `tensor.expand_shape` at a static result-dim
//     slot                                          -> static constant
//   * `tensor.dim` of `tensor.expand_shape` at a dynamic SSA slot
//                                                   -> the matching
//                                                      `output_shape`
//                                                      operand
//   * `tensor.dim` of `tensor.collapse_shape`       -> `affine.apply`
//                                                      product over
//                                                      reassociation
//                                                      group dims
//   * `tensor.dim` of a fully-static tensor          -> static constant
//   * Chained `dim(collapse(expand(arg)))`           -> single
//                                                      `affine.apply`
//                                                      on chain-root
//                                                      dims
//   * Function with no `tensor.dim` of a reshape op  -> bit-identical
//
// The pass uses upstream reify which lowers static factors INTO the
// affine map (e.g. group `(?, 32)` -> `affine_map<()[s0] -> (s0 * 32)>`)
// rather than emitting a separate `arith.muli %dyn, %c32`.

// -----

// CHECK-LABEL: @select_of_same_shape_value
// CHECK-NOT: arith.select
// CHECK: return %arg1
func.func @select_of_same_shape_value(%condition: i1, %extent: i64) -> i64 {
  %selected = arith.select %condition, %extent, %extent : i64
  return %selected : i64
}

// -----

// CHECK-LABEL: @dim_of_expand_static_slot
func.func @dim_of_expand_static_slot(%arg0: tensor<?x4096xf16>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %expand = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [%d0, 32, 128] : tensor<?x4096xf16> into tensor<?x32x128xf16>
  %use = tensor.dim %expand, %c1 : tensor<?x32x128xf16>
  return %use : index
}
// CHECK-NOT: tensor.expand_shape
// CHECK: arith.constant 32 : index
// CHECK: return

// -----

// CHECK-LABEL: @dim_of_expand_dynamic_slot
func.func @dim_of_expand_dynamic_slot(%arg0: tensor<?x4096xf16>) -> index {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %expand = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [%d0, 32, 128] : tensor<?x4096xf16> into tensor<?x32x128xf16>
  %use = tensor.dim %expand, %c0 : tensor<?x32x128xf16>
  return %use : index
}
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.dim %{{.*expanded}}
// CHECK: %[[D0:.+]] = tensor.dim %arg0, %{{.*}} : tensor<?x4096xf16>
// CHECK: return %[[D0]]

// -----

// CHECK-LABEL: @dim_of_collapse_two_dyn
func.func @dim_of_collapse_two_dyn(%arg0: tensor<?x?xf16>) -> index {
  %c0 = arith.constant 0 : index
  %collapse = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x?xf16> into tensor<?xf16>
  %use = tensor.dim %collapse, %c0 : tensor<?xf16>
  return %use : index
}
// CHECK-NOT: tensor.collapse_shape
// CHECK-DAG: tensor.dim %arg0
// CHECK: affine.apply
// CHECK: return

// -----

// Mixed static / dynamic source: the static factor is folded INTO the
// affine map by upstream reify -- no separate `arith.muli %dyn, %c32`.
// CHECK-LABEL: @dim_of_collapse_static_and_dyn
func.func @dim_of_collapse_static_and_dyn(%arg0: tensor<?x32xf16>) -> index {
  %c0 = arith.constant 0 : index
  %collapse = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x32xf16> into tensor<?xf16>
  %use = tensor.dim %collapse, %c0 : tensor<?xf16>
  return %use : index
}
// CHECK-NOT: tensor.collapse_shape
// CHECK: tensor.dim %arg0
// CHECK: affine.apply
// CHECK: return

// -----

// 3-source group: chain length tracks `|group|`.
// CHECK-LABEL: @dim_of_collapse_three_way
func.func @dim_of_collapse_three_way(%arg0: tensor<?x?x?xf16>) -> index {
  %c0 = arith.constant 0 : index
  %collapse = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<?x?x?xf16> into tensor<?xf16>
  %use = tensor.dim %collapse, %c0 : tensor<?xf16>
  return %use : index
}
// CHECK-NOT: tensor.collapse_shape
// CHECK-COUNT-3: tensor.dim %arg0
// CHECK: affine.apply
// CHECK: return

// -----

// Same-rank dynamic Reshape decomposed to expand_shape + collapse_shape:
// `dim(collapse, 1)` collapses end-to-end to `affine.apply [s0 * 32]` on
// `tensor.dim %arg0, 1`.  Canonical pattern for transformer decoder
// norm / projection chains.
// CHECK-LABEL: @same_rank_reshape_chain_dim
func.func @same_rank_reshape_chain_dim(%arg0: tensor<?x?x4096xf16>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x4096xf16>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x4096xf16>
  %expand = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [%d0, %d1, 32, 128] : tensor<?x?x4096xf16> into tensor<?x?x32x128xf16>
  %collapse = tensor.collapse_shape %expand [[0], [1, 2], [3]] : tensor<?x?x32x128xf16> into tensor<?x?x128xf16>
  %use = tensor.dim %collapse, %c1 : tensor<?x?x128xf16>
  return %use : index
}
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.dim %{{.*expanded}}
// CHECK-NOT: tensor.dim %{{.*collapsed}}
// CHECK: tensor.dim %arg0
// CHECK: affine.apply
// CHECK: return

// -----

// `tensor.dim` of a fully-static tensor folds to a constant via the
// `tensor.dim` canonicaliser (no reify needed).
// CHECK-LABEL: @dim_of_static_const
func.func @dim_of_static_const(%arg0: tensor<32x128xf16>) -> index {
  %c0 = arith.constant 0 : index
  %use = tensor.dim %arg0, %c0 : tensor<32x128xf16>
  return %use : index
}
// CHECK-NOT: tensor.dim
// CHECK: arith.constant 32 : index
// CHECK: return

// -----

// No-op: function with no `tensor.dim` of a reshape op should produce
// bit-identical IR.
// CHECK-LABEL: @no_dim_on_reshape_noop
func.func @no_dim_on_reshape_noop(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
  %empty = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  return %empty : tensor<?x?xf16>
}
// CHECK: tensor.dim %arg0, %c0
// CHECK: tensor.dim %arg0, %c1
// CHECK: tensor.empty
// CHECK: return

// -----

// No-op: collapse_shape live but never `tensor.dim`-queried -- preserved
// as an SSA value.
// CHECK-LABEL: @no_dim_on_collapse_noop
func.func @no_dim_on_collapse_noop(%arg0: tensor<?x?xf16>) -> tensor<?xf16> {
  %collapse = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x?xf16> into tensor<?xf16>
  return %collapse : tensor<?xf16>
}
// CHECK: tensor.collapse_shape
// CHECK: return
