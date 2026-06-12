// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// LpNormalization decomposes in the convert-onnx-to-hip pre-lowering loop
// into Mul(x,x) -> ReduceSum(axis, keepdims=1) -> Sqrt -> Div(x, norm). The
// final Div broadcasts (rhs has 1 along the reduce axis), so
// BroadcastDivToMulReciprocal -- also in the pre-lowering loop -- rewrites
// it into Mul(x, Reciprocal(norm)). The expected hip.* op set is therefore:
// hip.mul, hip.reduce_sum, hip.sqrt, hip.reciprocal, hip.mul. p=1 gets one
// extra hip.sqrt for the Sqrt(x*x)=|x| step.

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
  // CHECK: hip.mul(%[[CTX]]) ins(%[[X]], %[[X]]
  // CHECK: hip.reduce_sum
  // CHECK: hip.sqrt
  // CHECK: hip.reciprocal
  // CHECK: hip.mul

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
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: tensor.dim
  // CHECK: tensor.dim
  // CHECK: hip.mul
  // CHECK: hip.reduce_sum
  // CHECK: hip.sqrt
  // CHECK: hip.reciprocal
  // CHECK: hip.mul
}
