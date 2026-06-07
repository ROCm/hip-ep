// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt %s --hip-build-shape-fn -split-input-file | FileCheck %s
//
// Pins down the `@infer_shapes` function `--hip-build-shape-fn` emits: a pure
// index-arithmetic program mapping `@main_graph` input dims -> output dims.
// Args = an unused `!hip.context` arg 0 (module-wide invariant), then one
// `index` per (input tensor, dim) flattened over inputs (the !hip.context arg
// and non-tensor inputs contribute no dims); results = one `index` per (output
// tensor, dim). Coverage:
//
//   * non-identity B*S collapse  -> affine.apply product over input dims
//                                   + static dim as a constant
//   * identity output dim        -> the matching scalar arg, verbatim
//   * closed-form data-dependent -> tensor.extract on an input scalar becomes
//     output dim (Range-like)       an extra @infer_shapes VALUE arg, recorded
//                                   in the hipdnn.shape_fn_data_args attr
//   * non-resolvable data dim    -> kDynamic (INT64_MIN) sentinel (e.g. a
//                                   tensor.extract with a non-constant index);
//                                   the EP falls back to DimSource for that dim
//   * @main_graph itself is never mutated by the pass

// -----

// Non-identity: [B,S,4096] -> collapse -> [B*S, 4096]. dim0 is a product of
// two dynamic input dims (DimSource cannot express this). @infer_shapes carries
// an unused !hip.context arg 0; the input tensor's dims map to the scalar args
// that follow it.
func.func @main_graph(%ctx: !hip.context, %a: tensor<?x?x4096xf16>) -> tensor<?x4096xf16> {
  %collapse = tensor.collapse_shape %a [[0, 1], [2]] : tensor<?x?x4096xf16> into tensor<?x4096xf16>
  return %collapse : tensor<?x4096xf16>
}
// @main_graph is preserved verbatim (the pass only ADDS @infer_shapes).
// CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK-LABEL: func.func @main_graph
// CHECK: tensor.collapse_shape
//
// CHECK-LABEL: func.func @infer_shapes
// CHECK-SAME: (%{{[^:]+}}: !hip.context, %[[A0:[^:]+]]: index, %[[A1:[^:]+]]: index, %[[A2:[^:]+]]: index) -> (index, index)
// CHECK-DAG: %[[BS:.+]] = affine.apply #[[MAP]]()[%[[A0]], %[[A1]]]
// CHECK-DAG: %[[C4096:.+]] = arith.constant 4096 : index
// CHECK: return %[[BS]], %[[C4096]] : index, index

// -----

// Identity output dim (canonical LLM case): out dim0 == input dim0, out dim1
// is static. dim0 resolves straight to the scalar arg.
func.func @main_graph(%a: tensor<?x4096xf16>) -> tensor<?x4096xf16> {
  return %a : tensor<?x4096xf16>
}
// CHECK-LABEL: func.func @infer_shapes
// CHECK-SAME: (%{{[^:]+}}: !hip.context, %[[A0:[^:]+]]: index, %[[A1:[^:]+]]: index) -> (index, index)
// CHECK-DAG: %[[C4096:.+]] = arith.constant 4096 : index
// CHECK: return %[[A0]], %[[C4096]] : index, index

// -----

// Closed-form data-dependent (Range-like): the output dim is read from input
// CONTENTS via tensor.extract on a static-shape input with a constant index.
// The slice roots at the extract (a data leaf) -> @infer_shapes gains a VALUE
// arg (i64, after the input's dim args) and emits hipdnn.shape_fn_data_args.
func.func @main_graph(%a: tensor<4xi64>) -> tensor<?xf16> {
  %c0 = arith.constant 0 : index
  %v = tensor.extract %a[%c0] : tensor<4xi64>
  %d = arith.index_cast %v : i64 to index
  %e = tensor.empty(%d) : tensor<?xf16>
  return %e : tensor<?xf16>
}
// CHECK: hipdnn.shape_fn_data_args = [{elem_bits = 64 : i64, elem_offset = 0 : i64, input_idx = 0 : i64}]
// CHECK-LABEL: func.func @infer_shapes
// CHECK-SAME: (%{{[^:]+}}: !hip.context, %{{[^:]+}}: index, %[[V:[^:]+]]: i64) -> index
// CHECK: %[[D:.+]] = arith.index_cast %[[V]] : i64 to index
// CHECK: return %[[D]] : index

// -----

// Non-resolvable data dim: tensor.extract with a NON-constant index (the index
// itself comes from another input's value). The element offset is not
// statically known -> the extract is rejected as a leaf -> kDynamic sentinel,
// EP falls back to DimSource.
func.func @main_graph(%a: tensor<4xi64>, %b: tensor<i64>) -> tensor<?xf16> {
  %idx = tensor.extract %b[] : tensor<i64>
  %i = arith.index_cast %idx : i64 to index
  %v = tensor.extract %a[%i] : tensor<4xi64>
  %d = arith.index_cast %v : i64 to index
  %e = tensor.empty(%d) : tensor<?xf16>
  return %e : tensor<?xf16>
}
// CHECK-LABEL: func.func @infer_shapes
// CHECK: %[[K:.+]] = arith.constant -9223372036854775808 : index
// CHECK: return %[[K]] : index
