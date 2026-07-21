// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// LpNormalization has two lowering paths:
//
//   * Fused fast path (p=2, trailing normalize axis, STATIC trailing extent
//     N): emits a single SimplifiedLayerNormalization with scale=1/sqrt(N),
//     epsilon=0, which NormConversion lowers to one hip.rms_norm. This is the
//     q/k L2-norm shape in linear-attention decoders.
//
//   * Generic decomposition (everything else): Mul(x,x) -> ReduceSum -> Sqrt
//     -> Div(x, norm). The broadcasting Div is rewritten by
//     BroadcastDivToMulReciprocal into Mul(x, Reciprocal(norm)), so the hip.*
//     op set is hip.mul, hip.reduce_sum, hip.sqrt, hip.reciprocal, hip.mul.
//     p=1 gets one extra hip.sqrt for the Sqrt(x*x)=|x| step.

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Default p=2, axis=-1 on a static rank-2 f32 input. Trailing extent is
  // static (N=3) -> fused fast path -> a single hip.rms_norm.
  func.func @lp_norm_p2_default_f32(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %r = "onnx.LpNormalization"(%x) {axis = -1 : si64, p = 2 : si64}
        : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func @lp_norm_p2_default_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x3xf32>)
  // CHECK-NOT: onnx.LpNormalization
  // CHECK-NOT: hip.reduce_sum
  // CHECK: hip.rms_norm

  // p=1, explicit axis=1 on a rank-3 f16 input. p=1 never takes the fast path
  // (only p=2 maps to L2/RMS), so it decomposes. The p=1 path inserts an extra
  // Sqrt(x*x) before the reduction to compute |x|.
  func.func @lp_norm_p1_axis1_f16(%x: tensor<2x4x5xf16>) -> tensor<2x4x5xf16> {
    %r = "onnx.LpNormalization"(%x) {axis = 1 : si64, p = 1 : si64}
        : (tensor<2x4x5xf16>) -> tensor<2x4x5xf16>
    return %r : tensor<2x4x5xf16>
  }
  // CHECK-LABEL: func.func @lp_norm_p1_axis1_f16
  // CHECK-NOT: onnx.LpNormalization
  // CHECK-NOT: hip.rms_norm
  // p=1 path: Mul(x,x) -> Sqrt(=abs) -> ReduceSum -> ... -> Reciprocal -> Mul
  // CHECK: hip.mul
  // CHECK: hip.sqrt
  // CHECK: hip.reduce_sum
  // CHECK: hip.reciprocal
  // CHECK: hip.mul

  // p=2 over a NON-trailing axis (axis=1). The fast path requires the trailing
  // axis, so this falls back to the generic decomposition.
  func.func @lp_norm_p2_axis1_fallback(%x: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %r = "onnx.LpNormalization"(%x) {axis = 1 : si64, p = 2 : si64}
        : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %r : tensor<2x3x4xf32>
  }
  // CHECK-LABEL: func.func @lp_norm_p2_axis1_fallback
  // CHECK-NOT: onnx.LpNormalization
  // CHECK-NOT: hip.rms_norm
  // CHECK: hip.mul
  // CHECK: hip.reduce_sum
  // CHECK: hip.sqrt
  // CHECK: hip.reciprocal
  // CHECK: hip.mul

  // Dynamic batch + sequence dims, STATIC trailing feature dim. p=2 over the
  // trailing axis with a static extent (16) still fuses to a single
  // hip.rms_norm — dynamic outer dims do not block the fast path. This is the
  // canonical linear-attention q/k L2-norm shape.
  func.func @lp_norm_p2_dynamic_outer(%x: tensor<?x?x16xf32>) -> tensor<?x?x16xf32> {
    %r = "onnx.LpNormalization"(%x) {axis = -1 : si64, p = 2 : si64}
        : (tensor<?x?x16xf32>) -> tensor<?x?x16xf32>
    return %r : tensor<?x?x16xf32>
  }
  // CHECK-LABEL: func.func @lp_norm_p2_dynamic_outer
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[X2:.*]]: tensor<?x?x16xf32>)
  // CHECK-NOT: onnx.LpNormalization
  // CHECK-NOT: hip.reduce_sum
  // CHECK: hip.rms_norm
}
