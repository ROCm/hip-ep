// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// LpNormalization decomposes in the convert-onnx-to-hip pre-lowering loop
// into Mul(x,x) -> ReduceSum(axis, keepdims=1) -> Sqrt -> Div(x, norm). The
// final Div broadcasts (rhs has 1 along the reduce axis), so
// BroadcastDivToMulReciprocal -- also in the pre-lowering loop -- rewrites
// it into Mul(x, Reciprocal(norm)).
//
// For p=2 the resulting Mul/ReduceSum/Sqrt/Reciprocal/Mul chain is exactly
// the pattern L2NormFusion matches, so it is then folded into a single
// hip.l2_norm (epsilon = 0, numerically identical: x / sqrt(sum(x^2))).
// p=1 inserts an extra Sqrt(x*x)=|x| step, which does not match the L2
// pattern, so it stays as the primitive decomposition.

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Default p=2, axis=-1 on a static rank-2 f32 input.
  func.func @lp_norm_p2_default_f32(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %r = "onnx.LpNormalization"(%x) {axis = -1 : si64, p = 2 : si64}
        : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func @lp_norm_p2_default_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x3xf32>)
  // CHECK-NOT: onnx.LpNormalization
  // CHECK-NOT: onnx.Div
  // p=2 chain is folded into a single hip.l2_norm by L2NormFusion.
  // CHECK: hip.l2_norm(%[[CTX]]) ins(%[[X]]
  // CHECK-NOT: hip.reduce_sum

  // p=1, explicit axis=1 on a rank-3 f16 input. p=1 path inserts an extra
  // Sqrt(x*x) before the reduction to compute |x|.
  func.func @lp_norm_p1_axis1_f16(%x: tensor<2x4x5xf16>) -> tensor<2x4x5xf16> {
    %r = "onnx.LpNormalization"(%x) {axis = 1 : si64, p = 1 : si64}
        : (tensor<2x4x5xf16>) -> tensor<2x4x5xf16>
    return %r : tensor<2x4x5xf16>
  }
  // CHECK-LABEL: func.func @lp_norm_p1_axis1_f16
  // CHECK-NOT: onnx.LpNormalization
  // p=1 path: Mul(x,x) -> Sqrt(=abs) -> ReduceSum -> ... -> Reciprocal -> Mul
  // CHECK: hip.mul
  // CHECK: hip.sqrt
  // CHECK: hip.reduce_sum
  // CHECK: hip.reciprocal
  // CHECK: hip.mul

  // Dynamic batch + sequence dims; static feature dim along the reduce axis.
  // Verifies the decomposition handles dynamic shapes through the
  // tensor.dim + tensor.empty plumbing inherited from the primitive
  // converters (Mul / ReduceSum / Sqrt / Reciprocal).
  func.func @lp_norm_dynamic(%x: tensor<?x?x16xf32>) -> tensor<?x?x16xf32> {
    %r = "onnx.LpNormalization"(%x) {axis = -1 : si64, p = 2 : si64}
        : (tensor<?x?x16xf32>) -> tensor<?x?x16xf32>
    return %r : tensor<?x?x16xf32>
  }
  // CHECK-LABEL: func.func @lp_norm_dynamic
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[X2:.*]]: tensor<?x?x16xf32>)
  // CHECK-NOT: onnx.LpNormalization
  // p=2 dynamic-shape chain is likewise folded into hip.l2_norm.
  // CHECK: hip.l2_norm(%[[CTX2]]) ins(%[[X2]]
  // CHECK-NOT: hip.reduce_sum
}
