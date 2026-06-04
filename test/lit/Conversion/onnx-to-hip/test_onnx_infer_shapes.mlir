// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --onnx-infer-shapes --split-input-file %s | FileCheck %s

// Cases for `--onnx-infer-shapes`. Most test functions return `()` so
// the body op's pre-pass result type is decoupled from the function
// signature -- otherwise the parser would reject IR where the op
// declared `tensor<f16>` but the func said `-> tensor<2x3xf16>`. The
// `func_return_sync` case at the bottom is the one test where the
// pre-pass result type matches the declared return (both rank-0
// placeholder), so it can exercise the pass's signature-sync logic.

// -----

// Test 1: pointwise unary -- rank-0 placeholder promoted to operand
// rank. The Tanh rule derives the candidate `tensor<2x3xf16>` from
// operand[0]; meet vs the rank-0 existing type triggers rank promotion.
func.func @unary_rank0_promoted(%x: tensor<2x3xf16>) {
  %0 = "onnx.Tanh"(%x) : (tensor<2x3xf16>) -> tensor<f16>
  return
}
// CHECK-LABEL: func.func @unary_rank0_promoted
// CHECK: "onnx.Tanh"(%{{.*}}) : (tensor<2x3xf16>) -> tensor<2x3xf16>

// -----

// Test 2: same-rank dim meet preserves static dims from either side.
// Existing result is `tensor<?x3xf16>` (one static, one dynamic).
// Operand is `tensor<2x?xf16>` -- the rule produces a candidate
// `tensor<2x?xf16>`, meet yields `tensor<2x3xf16>` by combining the
// static dim from each side.
func.func @meet_preserves_static_dims(%x: tensor<2x?xf16>) {
  %0 = "onnx.Identity"(%x) : (tensor<2x?xf16>) -> tensor<?x3xf16>
  return
}
// CHECK-LABEL: func.func @meet_preserves_static_dims
// CHECK: "onnx.Identity"(%{{.*}}) : (tensor<2x?xf16>) -> tensor<2x3xf16>

// -----

// Test 3: pointwise broadcast -- rank-0 promotes to highest-rank
// operand. Canonical Loop-body case: `acc` is the rank-0 placeholder,
// the OTHER operand carries the real rank.
func.func @broadcast_rank0_to_rank3(%a: tensor<f16>, %b: tensor<2x3x4xf16>) {
  %0 = "onnx.Add"(%a, %b)
       : (tensor<f16>, tensor<2x3x4xf16>) -> tensor<f16>
  return
}
// CHECK-LABEL: func.func @broadcast_rank0_to_rank3
// CHECK: "onnx.Add"(%{{.*}}, %{{.*}})
// CHECK-SAME: -> tensor<2x3x4xf16>

// -----

// Test 4: comparison ops keep their i1 result element type. The
// broadcast rule reads element type from the EXISTING result type
// (here: i1), not from the operands (here: i32).
func.func @broadcast_comparison_keeps_i1(%a: tensor<2x3xi32>,
                                          %b: tensor<2x3xi32>) {
  %0 = "onnx.Equal"(%a, %b)
       : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<i1>
  return
}
// CHECK-LABEL: func.func @broadcast_comparison_keeps_i1
// CHECK: "onnx.Equal"(%{{.*}}, %{{.*}})
// CHECK-SAME: -> tensor<2x3xi1>

// -----

// Test 5: 3-operand broadcast -- `onnx.Where(cond, x, y)`. Rank-0 %x
// extends up to %y's rank-2.
func.func @where_three_operand_broadcast(%c: tensor<2x3xi1>,
                                          %x: tensor<f16>,
                                          %y: tensor<2x3xf16>) {
  %0 = "onnx.Where"(%c, %x, %y)
       : (tensor<2x3xi1>, tensor<f16>, tensor<2x3xf16>) -> tensor<f16>
  return
}
// CHECK-LABEL: func.func @where_three_operand_broadcast
// CHECK: "onnx.Where"(%{{.*}}, %{{.*}}, %{{.*}})
// CHECK-SAME: -> tensor<2x3xf16>

// -----

// Test 6: Concat axis dim summed; non-axis dim agreement preserved.
func.func @concat_axis_dim_summed(%a: tensor<2x3xf16>, %b: tensor<2x4xf16>) {
  %0 = "onnx.Concat"(%a, %b) {axis = 1 : si64}
       : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<f16>
  return
}
// CHECK-LABEL: func.func @concat_axis_dim_summed
// CHECK: "onnx.Concat"(%{{.*}}, %{{.*}})
// CHECK-SAME: -> tensor<2x7xf16>

// -----

// Test 7: Concat dynamic on the axis dim forces axis-dim-dynamic.
// shape[1] = ? + 4 -> ?; shape[0] agrees on 2.
func.func @concat_dynamic_axis_dim(%a: tensor<2x?xf16>, %b: tensor<2x4xf16>) {
  %0 = "onnx.Concat"(%a, %b) {axis = 1 : si64}
       : (tensor<2x?xf16>, tensor<2x4xf16>) -> tensor<f16>
  return
}
// CHECK-LABEL: func.func @concat_dynamic_axis_dim
// CHECK: "onnx.Concat"(%{{.*}}, %{{.*}})
// CHECK-SAME: -> tensor<2x?xf16>

// -----

// Test 8: Concat negative axis is wrap-corrected. axis = -1 -> 1
// (rank 2). Same outcome as axis = 1.
func.func @concat_negative_axis(%a: tensor<2x3xf16>, %b: tensor<2x4xf16>) {
  %0 = "onnx.Concat"(%a, %b) {axis = -1 : si64}
       : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<f16>
  return
}
// CHECK-LABEL: func.func @concat_negative_axis
// CHECK: "onnx.Concat"(%{{.*}}, %{{.*}})
// CHECK-SAME: -> tensor<2x7xf16>

// -----

// Test 9: Concat with rank-0 placeholder operand and ranked operand.
// Operand[0] is a rank-0 placeholder; operand[1] contributes rank 2.
// The axis dim is forced dynamic (we cannot sum a known value with
// the placeholder's unknown), but the rank is correct and non-axis
// dim 0 takes operand[1]'s 2.
func.func @concat_rank0_placeholder_promoted(%a: tensor<f16>,
                                              %b: tensor<2x3xf16>) {
  %0 = "onnx.Concat"(%a, %b) {axis = 1 : si64}
       : (tensor<f16>, tensor<2x3xf16>) -> tensor<f16>
  return
}
// CHECK-LABEL: func.func @concat_rank0_placeholder_promoted
// CHECK: "onnx.Concat"(%{{.*}}, %{{.*}})
// CHECK-SAME: -> tensor<2x?xf16>

// -----

// Test 10: Slice -- rank-preserving, all dims dynamic.
func.func @slice_all_dynamic(%x: tensor<2x3x4xf16>,
                              %s: tensor<3xi64>,
                              %e: tensor<3xi64>) {
  %0 = "onnx.Slice"(%x, %s, %e)
       : (tensor<2x3x4xf16>, tensor<3xi64>, tensor<3xi64>) -> tensor<f16>
  return
}
// CHECK-LABEL: func.func @slice_all_dynamic
// CHECK: "onnx.Slice"
// CHECK-SAME: -> tensor<?x?x?xf16>

// -----

// Test 11: LayerNormalization -- result 0 has operand[0]'s shape;
// results 1, 2 (mean / inv-std) are not handled (the rules library
// returns null for resultIdx > 0, so the cloned types persist).
// We exercise the single-result form here -- the multi-result form
// would simply leave indices 1 / 2 untouched.
func.func @layernorm_single_result(%x: tensor<2x3xf16>,
                                    %scale: tensor<3xf16>,
                                    %bias: tensor<3xf16>) {
  %0 = "onnx.LayerNormalization"(%x, %scale, %bias) {axis = -1 : si64,
                                                     epsilon = 9.99999974e-6 : f32}
       : (tensor<2x3xf16>, tensor<3xf16>, tensor<3xf16>) -> tensor<f16>
  return
}
// CHECK-LABEL: func.func @layernorm_single_result
// CHECK: "onnx.LayerNormalization"
// CHECK-SAME: -> tensor<2x3xf16>

// -----

// Test 12: SSA cascade -- a refined op's result propagates to its
// consumers in a single forward walk. The Add's rank-0 placeholder
// is refined to rank-3 (broadcast rule sees both operands as rank-3),
// then the Tanh consumer sees a rank-3 operand and its rule promotes
// the Tanh's rank-0 result to rank-3 too. No fixed-point iteration
// is needed.
func.func @cascade_forward_walk(%c: tensor<?x?x?xf16>) {
  %add = "onnx.Add"(%c, %c)
       : (tensor<?x?x?xf16>, tensor<?x?x?xf16>) -> tensor<f16>
  %tanh = "onnx.Tanh"(%add) : (tensor<f16>) -> tensor<f16>
  return
}
// CHECK-LABEL: func.func @cascade_forward_walk
// CHECK: %[[ADD:.*]] = "onnx.Add"
// CHECK-SAME: -> tensor<?x?x?xf16>
// CHECK: "onnx.Tanh"(%[[ADD]]) : (tensor<?x?x?xf16>) -> tensor<?x?x?xf16>

// -----

// Test 13: function return signature sync. The func is parseable at
// rank-0 (declared return == body op result == tensor<f16>); after
// the pass refines the body op result via the broadcast rule, the
// `func.return` operand type updates via SSA propagation, and
// `syncFuncReturnTypes` rewrites the declared return type to match.
//
// CHECK lines pin both the declared return type AND the body op
// result type post-pass.
func.func @func_return_sync(%a: tensor<f16>, %b: tensor<2x3x4xf16>)
    -> tensor<f16> {
  %0 = "onnx.Add"(%a, %b)
       : (tensor<f16>, tensor<2x3x4xf16>) -> tensor<f16>
  return %0 : tensor<f16>
}
// CHECK-LABEL: func.func @func_return_sync
// CHECK-SAME: -> tensor<2x3x4xf16>
// CHECK: %[[ADD:.*]] = "onnx.Add"
// CHECK-SAME: -> tensor<2x3x4xf16>
// CHECK: return %[[ADD]] : tensor<2x3x4xf16>

// -----

// Test 14: safety belt -- `onnx.ReduceSum` is intentionally absent
// from the rules library because it is a true rank-changing op
// (keepdims=0 over all axes goes from rank-3 to rank-0). The pass
// must NOT promote its rank-0 result back to rank-3, or the IR would
// lie about the result rank. The rules library returns null for
// ReduceSum, the pass skips the op, and the result type stays at
// `tensor<f32>`.
func.func @reduce_sum_stays_rank0(%x: tensor<?x?x?xf32>) {
  %sum = "onnx.ReduceSum"(%x) {keepdims = 0 : si64,
                                noop_with_empty_axes = 0 : si64}
       : (tensor<?x?x?xf32>) -> tensor<f32>
  return
}
// CHECK-LABEL: func.func @reduce_sum_stays_rank0
// CHECK: "onnx.ReduceSum"
// CHECK-SAME: -> tensor<f32>

// -----

// Test 15: safety belt -- `onnx.Reshape` reshapes its data operand
// according to its (typically constant) shape operand. The pass
// has no Reshape rule, so the rules library returns null and the
// result type is preserved verbatim.
func.func @reshape_preserved(%x: tensor<f16>, %newshape: tensor<4xi64>) {
  %y = "onnx.Reshape"(%x, %newshape) {allowzero = 0 : si64}
       : (tensor<f16>, tensor<4xi64>) -> tensor<1x?x16x72xf16>
  return
}
// CHECK-LABEL: func.func @reshape_preserved
// CHECK: "onnx.Reshape"
// CHECK-SAME: -> tensor<1x?x16x72xf16>

// -----

// Test 16: Cast -- result element type is preserved from the existing
// result type, NOT from operand[0]; only the shape is taken from
// operand[0]. This works because the importer always sets the right
// element type on the cloned op even when the shape is a rank-0
// placeholder.
func.func @cast_element_type_preserved(%x: tensor<2x3xf16>) {
  %0 = "onnx.Cast"(%x) {to = 1 : si64} : (tensor<2x3xf16>) -> tensor<f32>
  return
}
// CHECK-LABEL: func.func @cast_element_type_preserved
// CHECK: "onnx.Cast"
// CHECK-SAME: -> tensor<2x3xf32>
