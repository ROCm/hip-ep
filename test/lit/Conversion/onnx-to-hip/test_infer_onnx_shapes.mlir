// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the InferOnnxShapes pass tightens onnx.* op result types from loose
// (`<?x?x?>`) to the maximum static info derivable from the (still-symbolic)
// graph inputs, then propagates the tightening into the func.func return
// signature.
//
// Per-op rules under test:
//   * Reshape via Concat(Slice(Shape(x), …), const) — the canonical ViT
//     attention pattern. Static dims from the constant leg propagate; dyn
//     dims tracked through the Shape leg propagate when input dims are
//     static. ONNX `-1` is computed via element-count cancellation.
//   * Transpose with explicit `perm` and with default-reverse perm.
//   * MatMul outer-batch broadcast right-aligned over the OUTER slice only
//     (lhsRank-2 / rhsRank-2 window) — guards against the canonical
//     pitfall where right-aligning the WHOLE operand would silently trace
//     output dims to the K dimension.
//   * Cast preserves rank and dim sizes; element type swaps.
//   * Unary same-shape (Tanh / Softmax / Gelu / ...): result == input.
//   * Binary broadcast (Add / Sub / Mul / ...): numpy right-align.
//
// Refinement is invoked through the standalone --infer-onnx-shapes pass
// (lib/Conversion/OnnxToHip/InferOnnxShapes.cpp's pass wrapper). This is
// intentionally NOT routed through convert-onnx-to-hip so the test fails
// fast on a refinement regression rather than getting masked by a lowering
// error downstream.
// ============================================================================

// RUN: hip-mlir-opt --infer-onnx-shapes %s | FileCheck %s

// --- Reshape: const shape with all-positive dims ----------------------------
// Output dims should be tightened from `?x?x?x?` to `1x128x32x128`.
func.func @reshape_const_shape(%arg0: tensor<1x128x4096xf16>)
    -> tensor<?x?x?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<[1, 128, 32, 128]> : tensor<4xi64>}
      : () -> tensor<4xi64>
  %r = "onnx.Reshape"(%arg0, %shape)
      : (tensor<1x128x4096xf16>, tensor<4xi64>) -> tensor<?x?x?x?xf16>
  return %r : tensor<?x?x?x?xf16>
}
// CHECK-LABEL: func.func @reshape_const_shape
// CHECK-SAME: -> tensor<1x128x32x128xf16>
// CHECK: return {{.*}} : tensor<1x128x32x128xf16>

// --- Reshape: ONNX -1 inference via element-count cancellation -------------
// shape = [1, 128, -1], input element count = 1*128*4096; -1 dim = 4096.
func.func @reshape_minus_one(%arg0: tensor<1x128x32x128xf16>)
    -> tensor<?x?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<[1, 128, -1]> : tensor<3xi64>}
      : () -> tensor<3xi64>
  %r = "onnx.Reshape"(%arg0, %shape)
      : (tensor<1x128x32x128xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
  return %r : tensor<?x?x?xf16>
}
// CHECK-LABEL: func.func @reshape_minus_one
// CHECK-SAME: -> tensor<1x128x4096xf16>

// --- Reshape: Qwen-style patch-merger (divide dim 0 by K) -----------------
// shape = [-1, 4608] on input <num_patches x 1152>. ONNX `-1` infers the
// missing slot from element-count cancellation: output_dim0 * 4608 ==
// num_patches * 1152, so output_dim0 = num_patches / 4. The forward
// refinement here only sees dim 1 = 4608; the divide-by-K story is the
// SSA-trace side (verified end-to-end by test/python/test_qwen3_5_9b.py).
// This case guards the inputDynCount == resolvedDynCount path so a
// regression that disables -1 inference for ≠1 ratios surfaces in LIT.
func.func @reshape_qwen_patch_merger(%x: tensor<?x1152xf16>)
    -> tensor<?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<[-1, 4608]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %r = "onnx.Reshape"(%x, %shape)
      : (tensor<?x1152xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  return %r : tensor<?x?xf16>
}
// CHECK-LABEL: func.func @reshape_qwen_patch_merger
// CHECK-SAME: -> tensor<?x4608xf16>

// --- Reshape: dyn batch through Shape(x) leg + static tail const -----------
// shape = Concat(Slice(Shape(x), 0..1), [256, 1152]) where x is <?x1152x16x16>.
// Output dim 0 traces back to x's dim 0 (still dynamic), dims 1+2 are static.
func.func @reshape_dyn_batch_through_shape(%x: tensor<?x1152x16x16xf16>)
    -> tensor<?x?x?xf16> {
  %shape_x = "onnx.Shape"(%x) : (tensor<?x1152x16x16xf16>) -> tensor<4xi64>
  %s0 = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>}
      : () -> tensor<1xi64>
  %e0 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>}
      : () -> tensor<1xi64>
  %a0 = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>}
      : () -> tensor<1xi64>
  %head = "onnx.Slice"(%shape_x, %s0, %e0, %a0)
      : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<1xi64>
  %tail = "onnx.Constant"() {value = dense<[256, 1152]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %shape = "onnx.Concat"(%head, %tail) {axis = 0 : si64}
      : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
  %r = "onnx.Reshape"(%x, %shape)
      : (tensor<?x1152x16x16xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
  return %r : tensor<?x?x?xf16>
}
// CHECK-LABEL: func.func @reshape_dyn_batch_through_shape
// CHECK-SAME: -> tensor<?x256x1152xf16>

// --- Transpose: explicit perm = [0, 2, 1] -----------------------------------
// Input <?x4x8xf32>; output via perm becomes <?x8x4xf32>.
func.func @transpose_explicit_perm(%arg0: tensor<?x4x8xf32>)
    -> tensor<?x?x?xf32> {
  %r = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]}
      : (tensor<?x4x8xf32>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}
// CHECK-LABEL: func.func @transpose_explicit_perm
// CHECK-SAME: -> tensor<?x8x4xf32>

// --- Transpose: default perm (reverse) --------------------------------------
// Input <2x3x4xf32>; output is <4x3x2xf32>.
func.func @transpose_default_perm(%arg0: tensor<2x3x4xf32>)
    -> tensor<?x?x?xf32> {
  %r = "onnx.Transpose"(%arg0) : (tensor<2x3x4xf32>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}
// CHECK-LABEL: func.func @transpose_default_perm
// CHECK-SAME: -> tensor<4x3x2xf32>

// --- MatMul: outer-batch right-aligned over the OUTER slice only -----------
// lhs <?x4x8xf32> (outer batch is `?`), rhs <8x16xf32> (no outer batch ->
// implicit broadcast 1). Out outer batch = ?; M = 4 from lhs[-2]; N = 16 from
// rhs[-1]. THIS IS THE GOTCHA TEST — right-aligning over the full operand
// would land lhsIdx on the K dim (8) and silently mis-trace the batch.
func.func @matmul_outer_batch_alignment(%lhs: tensor<?x4x8xf32>,
                                         %rhs: tensor<8x16xf32>)
    -> tensor<?x?x?xf32> {
  %r = "onnx.MatMul"(%lhs, %rhs)
      : (tensor<?x4x8xf32>, tensor<8x16xf32>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}
// CHECK-LABEL: func.func @matmul_outer_batch_alignment
// CHECK-SAME: -> tensor<?x4x16xf32>

// --- MatMul: both operands fully static -------------------------------------
func.func @matmul_static(%lhs: tensor<1x128x4096xf32>,
                          %rhs: tensor<4096x1024xf32>)
    -> tensor<?x?x?xf32> {
  %r = "onnx.MatMul"(%lhs, %rhs)
      : (tensor<1x128x4096xf32>, tensor<4096x1024xf32>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}
// CHECK-LABEL: func.func @matmul_static
// CHECK-SAME: -> tensor<1x128x1024xf32>

// --- Cast: preserves shape, swaps element type ------------------------------
func.func @cast_preserves_shape(%arg0: tensor<?x4x8xf32>)
    -> tensor<?x?x?xf16> {
  %r = "onnx.Cast"(%arg0) {to = 10 : si64}
      : (tensor<?x4x8xf32>) -> tensor<?x?x?xf16>
  return %r : tensor<?x?x?xf16>
}
// CHECK-LABEL: func.func @cast_preserves_shape
// CHECK-SAME: -> tensor<?x4x8xf16>

// --- Unary same-shape: chain of Tanh + Sigmoid + Gelu through dyn batch ----
func.func @unary_same_shape_chain(%arg0: tensor<?x256x1152xf32>)
    -> tensor<?x?x?xf32> {
  %t = "onnx.Tanh"(%arg0)
      : (tensor<?x256x1152xf32>) -> tensor<?x?x?xf32>
  %s = "onnx.Sigmoid"(%t)
      : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %g = "onnx.Gelu"(%s) {approximate = "tanh"}
      : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %g : tensor<?x?x?xf32>
}
// CHECK-LABEL: func.func @unary_same_shape_chain
// CHECK-SAME: -> tensor<?x256x1152xf32>

// --- Binary broadcast: right-aligned numpy semantics -----------------------
// lhs <?x4xf32>, rhs <8xf32> -> <?x?> tightens to <?x{max(4,8)}> via
// broadcast rule (the static-1-on-left case: dim 4 vs dim 8 -> 8 because
// the lhs's rightmost is 4 but the rhs has no left dim -> implicit 1
// broadcasts UP to 4? No — see numpy: rhs <8> right-aligns to lhs's
// rightmost; both must be compatible. 4 != 8 with neither 1, so this is
// invalid; use 4 instead for the broadcast test.
func.func @binary_broadcast(%lhs: tensor<?x4xf32>, %rhs: tensor<4xf32>)
    -> tensor<?x?xf32> {
  %r = "onnx.Add"(%lhs, %rhs)
      : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @binary_broadcast
// CHECK-SAME: -> tensor<?x4xf32>

// --- Binary broadcast: rank promotion + size-1 broadcast -------------------
// lhs <1x8xf32>, rhs <?x1xf32> -> output <?x8xf32> (dim 0 from rhs, dim 1
// from lhs by broadcast).
func.func @binary_broadcast_rank_promote(%lhs: tensor<1x8xf32>,
                                          %rhs: tensor<?x1xf32>)
    -> tensor<?x?xf32> {
  %r = "onnx.Mul"(%lhs, %rhs)
      : (tensor<1x8xf32>, tensor<?x1xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @binary_broadcast_rank_promote
// CHECK-SAME: -> tensor<?x8xf32>
