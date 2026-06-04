// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-test-onnx-result-type-inference --verify-diagnostics

// Per-rule unit tests for the `OnnxResultTypeInferenceInterface` rules
// library. Every assertion is via `expected-remark@+1` directives
// consumed by `--verify-diagnostics`; no FileCheck pipe is needed.
//
// Each test function returns `()` (void) so the SSA value of the op
// under test can carry any declared result type independent of the
// function's return signature -- the interface does not modify IR,
// it only returns the inferred type for the remark, so the SSA value's
// declared type can be the rank-0 placeholder we are checking the rule
// will replace.
//
// What these cases pin per group:
//   1. Pointwise unary -- result shape from operand[0], element type
//      preserved from the existing result type (so Cast / CastLike
//      keep their post-cast element type without inspecting the `to`
//      attribute). Stale rank-0 placeholders get promoted to the
//      operand's rank.
//   2. Pointwise broadcast -- numpy-style align-right broadcast over
//      all operands, comparison ops keep their i1 result element type.
//      Rank-0 placeholders promote to the highest-rank operand.
//   3. Concat -- axis dim is the sum of operand axis dims, non-axis
//      dims agree-or-dynamic, axis attr is wrap-corrected for negative
//      values.
//   4. Slice -- rank-preserving, all dims dynamic.
//   5. LayerNormalization -- result 0 has operand[0] shape; results
//      1 / 2 (mean / inv-std) intentionally unhandled.
//   6. Negative -- ops outside the rules library still go through the
//      dispatch path (no `no_iface` remark) and return the
//      `<unhandled>` sentinel.

// -----------------------------------------------------------------------------
// 1. Pointwise unary
// -----------------------------------------------------------------------------

func.func @unary_identity(%x: tensor<2x3xf16>) {
  // Stale rank-0 placeholder result -- the rule pulls the rank back
  // from operand[0].
  // expected-remark@+1 {{onnx.Identity computed: 'tensor<2x3xf16>'}}
  %0 = "onnx.Identity"(%x) : (tensor<2x3xf16>) -> tensor<f16>
  return
}

func.func @unary_tanh_dynamic(%x: tensor<?x?xf16>) {
  // Dynamic dims propagate as kDynamic in the inferred type.
  // expected-remark@+1 {{onnx.Tanh computed: 'tensor<?x?xf16>'}}
  %0 = "onnx.Tanh"(%x) : (tensor<?x?xf16>) -> tensor<f16>
  return
}

func.func @unary_cast_element_type_preserved(%x: tensor<2x3xf16>) {
  // Cast result element type comes from the existing (cloned) result
  // type, NOT from operand[0]; only the shape is taken from operand[0].
  // expected-remark@+1 {{onnx.Cast computed: 'tensor<2x3xf32>'}}
  %0 = "onnx.Cast"(%x) {to = 1 : si64} : (tensor<2x3xf16>) -> tensor<f32>
  return
}

// -----------------------------------------------------------------------------
// 2. Pointwise broadcast
// -----------------------------------------------------------------------------

func.func @broadcast_same_shape(%a: tensor<2x3xf16>, %b: tensor<2x3xf16>) {
  // expected-remark@+1 {{onnx.Add computed: 'tensor<2x3xf16>'}}
  %0 = "onnx.Add"(%a, %b)
      : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  return
}

func.func @broadcast_rank0_to_rank3(%a: tensor<f16>, %b: tensor<2x3x4xf16>) {
  // Canonical Loop-body case: operand[0] is a rank-0 placeholder, the
  // OTHER operand carries the real rank. Broadcast picks max rank and
  // produces the concrete shape from operand[1].
  // expected-remark@+1 {{onnx.Add computed: 'tensor<2x3x4xf16>'}}
  %0 = "onnx.Add"(%a, %b)
      : (tensor<f16>, tensor<2x3x4xf16>) -> tensor<f16>
  return
}

func.func @broadcast_scalar_extension(%a: tensor<2x3xf16>, %b: tensor<3xf16>) {
  // Right-aligned broadcast: rank-1 (3) extends to rank-2 (2x3).
  // expected-remark@+1 {{onnx.Mul computed: 'tensor<2x3xf16>'}}
  %0 = "onnx.Mul"(%a, %b)
      : (tensor<2x3xf16>, tensor<3xf16>) -> tensor<f16>
  return
}

func.func @broadcast_dynamic_propagation(%a: tensor<?x3xf16>,
                                          %b: tensor<2x3xf16>) {
  // Dynamic on the leading dim of %a forces the broadcast to keep
  // that dim dynamic, even though %b has a concrete 2.
  // expected-remark@+1 {{onnx.Sub computed: 'tensor<?x3xf16>'}}
  %0 = "onnx.Sub"(%a, %b)
      : (tensor<?x3xf16>, tensor<2x3xf16>) -> tensor<f16>
  return
}

func.func @broadcast_comparison_keeps_i1(%a: tensor<2x3xi32>,
                                          %b: tensor<2x3xi32>) {
  // Comparison ops -- result element type (i1) preserved from the
  // existing result type, not derived from the operands.
  // expected-remark@+1 {{onnx.Equal computed: 'tensor<2x3xi1>'}}
  %0 = "onnx.Equal"(%a, %b)
      : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<i1>
  return
}

func.func @broadcast_where_three_operand(%c: tensor<2x3xi1>,
                                          %x: tensor<f16>,
                                          %y: tensor<2x3xf16>) {
  // Where takes (cond, x, y) -- 3-operand broadcast; rank-0 %x extends
  // up to %y's rank-2.
  // expected-remark@+1 {{onnx.Where computed: 'tensor<2x3xf16>'}}
  %0 = "onnx.Where"(%c, %x, %y)
      : (tensor<2x3xi1>, tensor<f16>, tensor<2x3xf16>) -> tensor<f16>
  return
}

// -----------------------------------------------------------------------------
// 3. Concat
// -----------------------------------------------------------------------------

func.func @concat_axis_dim_summed(%a: tensor<2x3xf16>, %b: tensor<2x4xf16>) {
  // axis = 1 -> shape[1] = 3 + 4 = 7; shape[0] agrees on 2.
  // expected-remark@+1 {{onnx.Concat computed: 'tensor<2x7xf16>'}}
  %0 = "onnx.Concat"(%a, %b) {axis = 1 : si64}
      : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<f16>
  return
}

func.func @concat_dynamic_axis_dim(%a: tensor<2x?xf16>, %b: tensor<2x4xf16>) {
  // ANY dynamic operand on the axis dim forces the result axis dim
  // dynamic.
  // expected-remark@+1 {{onnx.Concat computed: 'tensor<2x?xf16>'}}
  %0 = "onnx.Concat"(%a, %b) {axis = 1 : si64}
      : (tensor<2x?xf16>, tensor<2x4xf16>) -> tensor<f16>
  return
}

func.func @concat_negative_axis(%a: tensor<2x3xf16>, %b: tensor<2x4xf16>) {
  // axis = -1 -> 1 (rank 2). Same outcome as axis = 1.
  // expected-remark@+1 {{onnx.Concat computed: 'tensor<2x7xf16>'}}
  %0 = "onnx.Concat"(%a, %b) {axis = -1 : si64}
      : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<f16>
  return
}

func.func @concat_rank0_placeholder_promoted(%a: tensor<f16>,
                                              %b: tensor<2x3xf16>) {
  // Operand[0] is a rank-0 placeholder; the ranked operand[1]
  // contributes rank 2. Axis dim is dynamic (we cannot sum a known
  // value with the placeholder's unknown), but the rank is correct
  // and non-axis dim 0 takes operand[1]'s 2.
  // expected-remark@+1 {{onnx.Concat computed: 'tensor<2x?xf16>'}}
  %0 = "onnx.Concat"(%a, %b) {axis = 1 : si64}
      : (tensor<f16>, tensor<2x3xf16>) -> tensor<f16>
  return
}

// -----------------------------------------------------------------------------
// 4. Slice (conservative: rank-preserving, all dims dynamic)
// -----------------------------------------------------------------------------

func.func @slice_all_dynamic(%x: tensor<2x3x4xf16>,
                              %s: tensor<3xi64>,
                              %e: tensor<3xi64>) {
  // expected-remark@+1 {{onnx.Slice computed: 'tensor<?x?x?xf16>'}}
  %0 = "onnx.Slice"(%x, %s, %e)
      : (tensor<2x3x4xf16>, tensor<3xi64>, tensor<3xi64>) -> tensor<f16>
  return
}

// -----------------------------------------------------------------------------
// 5. LayerNormalization
// -----------------------------------------------------------------------------

func.func @layernorm_single_result(%x: tensor<2x3xf16>,
                                    %scale: tensor<3xf16>,
                                    %bias: tensor<3xf16>) {
  // expected-remark@+1 {{onnx.LayerNormalization computed: 'tensor<2x3xf16>'}}
  %0 = "onnx.LayerNormalization"(%x, %scale, %bias) {axis = -1 : si64,
                                                     epsilon = 9.99999974e-6 : f32}
      : (tensor<2x3xf16>, tensor<3xf16>, tensor<3xf16>) -> tensor<f16>
  return
}

// -----------------------------------------------------------------------------
// 6. Negative -- ops outside the rules library still hit the dispatch
//    (no `no_iface` remark) but return the `<unhandled>` sentinel.
// -----------------------------------------------------------------------------

func.func @unhandled_op(%x: tensor<2x3xf16>) {
  // expected-remark@+1 {{onnx.UnknownOp computed: <unhandled>}}
  %0 = "onnx.UnknownOp"(%x) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  return
}

// Also keep a positive case for `Identity` + `Add` from commit 3 to
// guard against a regression where the dispatch reaches the FallbackModel
// but the rule lookup is silently broken (e.g. switch-on-name typo).
func.func @dispatch_baseline_post_rules(%a: tensor<2x3xf16>,
                                         %b: tensor<2x3xf16>) {
  // expected-remark@+1 {{onnx.Add computed: 'tensor<2x3xf16>'}}
  %0 = "onnx.Add"(%a, %b)
      : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  // expected-remark@+1 {{onnx.Identity computed: 'tensor<2x3xf16>'}}
  %1 = "onnx.Identity"(%0) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  return
}
