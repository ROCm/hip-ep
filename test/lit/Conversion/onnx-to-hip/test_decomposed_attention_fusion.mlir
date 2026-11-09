// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the pre-lowering DecomposedAttentionFusion folds primitive
// scaled dot-product attention (head-split Reshape/Transpose, Q @ K^T,
// optional scalar scales, Softmax, P @ V, head merge) into
// hip.multi_head_attention, and leaves chains it cannot prove equivalent alone.
//
// Test cases:
// 1. Self-attention with the scale split over Q and K (diffusers UNet export)
// 2. Cross-attention: key/value sequence differs from the query's
// 3. Scale applied to the scores with a Div, no Q/K scale
// 4. No scale at all: the fused op gets an explicit scale of 1
// 5. Negative: an additive mask between the GEMM and the Softmax
// 6. Negative: the probabilities have a second user
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%q: tensor<2x64x320xf16>) -> tensor<2x64x320xf16> {
    return %q : tensor<2x64x320xf16>
  }

  // CHECK-LABEL: func.func @split_scale
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[Q:.*]]: tensor<2x64x320xf16>, %[[K:.*]]: tensor<2x64x320xf16>, %[[V:.*]]: tensor<2x64x320xf16>)
  // CHECK: %[[A:.*]] = hip.multi_head_attention(%[[CTX]]) ins(%[[Q]], %[[K]], %[[V]]
  // CHECK-SAME: num_heads = 8
  // CHECK-SAME: scale = 0.15{{[0-9]+}} : f32
  // CHECK: tensor.expand_shape %[[A]] {{\[}}[0], [1], [2, 3]] output_shape [2, 64, 8, 40] : tensor<2x64x320xf16> into tensor<2x64x8x40xf16>
  // CHECK-NOT: softmax
  // CHECK-NOT: onnx.Softmax
  func.func @split_scale(%q: tensor<2x64x320xf16>, %k: tensor<2x64x320xf16>,
                         %v: tensor<2x64x320xf16>) -> tensor<128x320xf16> {
    %split = "onnx.Constant"() {value = dense<[2, -1, 8, 40]> : tensor<4xi64>} : () -> tensor<4xi64>
    %merge = "onnx.Constant"() {value = dense<[128, 320]> : tensor<2xi64>} : () -> tensor<2xi64>
    %c = "onnx.Constant"() {value = dense<3.977050e-01> : tensor<f16>} : () -> tensor<f16>
    %q4 = "onnx.Reshape"(%q, %split) : (tensor<2x64x320xf16>, tensor<4xi64>) -> tensor<2x64x8x40xf16>
    %qt = "onnx.Transpose"(%q4) {perm = [0, 2, 1, 3]} : (tensor<2x64x8x40xf16>) -> tensor<2x8x64x40xf16>
    %qs = "onnx.Mul"(%qt, %c) : (tensor<2x8x64x40xf16>, tensor<f16>) -> tensor<2x8x64x40xf16>
    %k4 = "onnx.Reshape"(%k, %split) : (tensor<2x64x320xf16>, tensor<4xi64>) -> tensor<2x64x8x40xf16>
    %kt = "onnx.Transpose"(%k4) {perm = [0, 2, 3, 1]} : (tensor<2x64x8x40xf16>) -> tensor<2x8x40x64xf16>
    %ks = "onnx.Mul"(%kt, %c) : (tensor<2x8x40x64xf16>, tensor<f16>) -> tensor<2x8x40x64xf16>
    %s = "onnx.MatMul"(%qs, %ks) : (tensor<2x8x64x40xf16>, tensor<2x8x40x64xf16>) -> tensor<2x8x64x64xf16>
    %p = "onnx.Softmax"(%s) {axis = -1 : si64} : (tensor<2x8x64x64xf16>) -> tensor<2x8x64x64xf16>
    %v4 = "onnx.Reshape"(%v, %split) : (tensor<2x64x320xf16>, tensor<4xi64>) -> tensor<2x64x8x40xf16>
    %vt = "onnx.Transpose"(%v4) {perm = [0, 2, 1, 3]} : (tensor<2x64x8x40xf16>) -> tensor<2x8x64x40xf16>
    %o = "onnx.MatMul"(%p, %vt) : (tensor<2x8x64x64xf16>, tensor<2x8x64x40xf16>) -> tensor<2x8x64x40xf16>
    %ot = "onnx.Transpose"(%o) {perm = [0, 2, 1, 3]} : (tensor<2x8x64x40xf16>) -> tensor<2x64x8x40xf16>
    %y = "onnx.Reshape"(%ot, %merge) : (tensor<2x64x8x40xf16>, tensor<2xi64>) -> tensor<128x320xf16>
    return %y : tensor<128x320xf16>
  }

  // CHECK-LABEL: func.func @cross_attention
  // CHECK: hip.multi_head_attention(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x64x320xf16>, tensor<2x77x320xf16>, tensor<2x77x320xf16>)
  // CHECK-SAME: num_heads = 8
  // CHECK-NOT: softmax
  func.func @cross_attention(%q: tensor<2x64x320xf16>, %k: tensor<2x77x320xf16>,
                             %v: tensor<2x77x320xf16>) -> tensor<2x64x8x40xf16> {
    %split = "onnx.Constant"() {value = dense<[2, -1, 8, 40]> : tensor<4xi64>} : () -> tensor<4xi64>
    %c = "onnx.Constant"() {value = dense<3.977050e-01> : tensor<f16>} : () -> tensor<f16>
    %q4 = "onnx.Reshape"(%q, %split) : (tensor<2x64x320xf16>, tensor<4xi64>) -> tensor<2x64x8x40xf16>
    %qt = "onnx.Transpose"(%q4) {perm = [0, 2, 1, 3]} : (tensor<2x64x8x40xf16>) -> tensor<2x8x64x40xf16>
    %qs = "onnx.Mul"(%qt, %c) : (tensor<2x8x64x40xf16>, tensor<f16>) -> tensor<2x8x64x40xf16>
    %k4 = "onnx.Reshape"(%k, %split) : (tensor<2x77x320xf16>, tensor<4xi64>) -> tensor<2x77x8x40xf16>
    %kt = "onnx.Transpose"(%k4) {perm = [0, 2, 3, 1]} : (tensor<2x77x8x40xf16>) -> tensor<2x8x40x77xf16>
    %ks = "onnx.Mul"(%kt, %c) : (tensor<2x8x40x77xf16>, tensor<f16>) -> tensor<2x8x40x77xf16>
    %s = "onnx.MatMul"(%qs, %ks) : (tensor<2x8x64x40xf16>, tensor<2x8x40x77xf16>) -> tensor<2x8x64x77xf16>
    %p = "onnx.Softmax"(%s) {axis = -1 : si64} : (tensor<2x8x64x77xf16>) -> tensor<2x8x64x77xf16>
    %v4 = "onnx.Reshape"(%v, %split) : (tensor<2x77x320xf16>, tensor<4xi64>) -> tensor<2x77x8x40xf16>
    %vt = "onnx.Transpose"(%v4) {perm = [0, 2, 1, 3]} : (tensor<2x77x8x40xf16>) -> tensor<2x8x77x40xf16>
    %o = "onnx.MatMul"(%p, %vt) : (tensor<2x8x64x77xf16>, tensor<2x8x77x40xf16>) -> tensor<2x8x64x40xf16>
    %ot = "onnx.Transpose"(%o) {perm = [0, 2, 1, 3]} : (tensor<2x8x64x40xf16>) -> tensor<2x64x8x40xf16>
    return %ot : tensor<2x64x8x40xf16>
  }

  // CHECK-LABEL: func.func @score_div
  // CHECK: hip.multi_head_attention
  // CHECK-SAME: num_heads = 4
  // CHECK-SAME: scale = 1.250000e-01 : f32
  // CHECK-NOT: softmax
  func.func @score_div(%q: tensor<1x32x256xf16>, %k: tensor<1x32x256xf16>,
                       %v: tensor<1x32x256xf16>) -> tensor<1x32x4x64xf16> {
    %split = "onnx.Constant"() {value = dense<[1, 32, 4, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
    %d = "onnx.Constant"() {value = dense<8.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %q4 = "onnx.Reshape"(%q, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %qt = "onnx.Transpose"(%q4) {perm = [0, 2, 1, 3]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x32x64xf16>
    %k4 = "onnx.Reshape"(%k, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %kt = "onnx.Transpose"(%k4) {perm = [0, 2, 3, 1]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x64x32xf16>
    %s = "onnx.MatMul"(%qt, %kt) : (tensor<1x4x32x64xf16>, tensor<1x4x64x32xf16>) -> tensor<1x4x32x32xf16>
    %ss = "onnx.Div"(%s, %d) : (tensor<1x4x32x32xf16>, tensor<f16>) -> tensor<1x4x32x32xf16>
    %p = "onnx.Softmax"(%ss) {axis = 3 : si64} : (tensor<1x4x32x32xf16>) -> tensor<1x4x32x32xf16>
    %v4 = "onnx.Reshape"(%v, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %vt = "onnx.Transpose"(%v4) {perm = [0, 2, 1, 3]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x32x64xf16>
    %o = "onnx.MatMul"(%p, %vt) : (tensor<1x4x32x32xf16>, tensor<1x4x32x64xf16>) -> tensor<1x4x32x64xf16>
    %ot = "onnx.Transpose"(%o) {perm = [0, 2, 1, 3]} : (tensor<1x4x32x64xf16>) -> tensor<1x32x4x64xf16>
    return %ot : tensor<1x32x4x64xf16>
  }

  // CHECK-LABEL: func.func @no_scale
  // CHECK: hip.multi_head_attention
  // CHECK-SAME: scale = 1.000000e+00 : f32
  func.func @no_scale(%q: tensor<1x32x256xf16>, %k: tensor<1x32x256xf16>,
                      %v: tensor<1x32x256xf16>) -> tensor<1x32x4x64xf16> {
    %split = "onnx.Constant"() {value = dense<[1, 32, 4, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
    %q4 = "onnx.Reshape"(%q, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %qt = "onnx.Transpose"(%q4) {perm = [0, 2, 1, 3]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x32x64xf16>
    %k4 = "onnx.Reshape"(%k, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %kt = "onnx.Transpose"(%k4) {perm = [0, 2, 3, 1]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x64x32xf16>
    %s = "onnx.MatMul"(%qt, %kt) : (tensor<1x4x32x64xf16>, tensor<1x4x64x32xf16>) -> tensor<1x4x32x32xf16>
    %p = "onnx.Softmax"(%s) {axis = -1 : si64} : (tensor<1x4x32x32xf16>) -> tensor<1x4x32x32xf16>
    %v4 = "onnx.Reshape"(%v, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %vt = "onnx.Transpose"(%v4) {perm = [0, 2, 1, 3]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x32x64xf16>
    %o = "onnx.MatMul"(%p, %vt) : (tensor<1x4x32x32xf16>, tensor<1x4x32x64xf16>) -> tensor<1x4x32x64xf16>
    %ot = "onnx.Transpose"(%o) {perm = [0, 2, 1, 3]} : (tensor<1x4x32x64xf16>) -> tensor<1x32x4x64xf16>
    return %ot : tensor<1x32x4x64xf16>
  }

  // CHECK-LABEL: func.func @masked
  // CHECK-NOT: hip.multi_head_attention
  // CHECK: hip.miopen.softmax
  func.func @masked(%q: tensor<1x32x256xf16>, %k: tensor<1x32x256xf16>,
                    %v: tensor<1x32x256xf16>, %mask: tensor<1x1x32x32xf16>)
      -> tensor<1x32x4x64xf16> {
    %split = "onnx.Constant"() {value = dense<[1, 32, 4, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
    %q4 = "onnx.Reshape"(%q, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %qt = "onnx.Transpose"(%q4) {perm = [0, 2, 1, 3]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x32x64xf16>
    %k4 = "onnx.Reshape"(%k, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %kt = "onnx.Transpose"(%k4) {perm = [0, 2, 3, 1]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x64x32xf16>
    %s = "onnx.MatMul"(%qt, %kt) : (tensor<1x4x32x64xf16>, tensor<1x4x64x32xf16>) -> tensor<1x4x32x32xf16>
    %sm = "onnx.Add"(%s, %mask) : (tensor<1x4x32x32xf16>, tensor<1x1x32x32xf16>) -> tensor<1x4x32x32xf16>
    %p = "onnx.Softmax"(%sm) {axis = -1 : si64} : (tensor<1x4x32x32xf16>) -> tensor<1x4x32x32xf16>
    %v4 = "onnx.Reshape"(%v, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %vt = "onnx.Transpose"(%v4) {perm = [0, 2, 1, 3]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x32x64xf16>
    %o = "onnx.MatMul"(%p, %vt) : (tensor<1x4x32x32xf16>, tensor<1x4x32x64xf16>) -> tensor<1x4x32x64xf16>
    %ot = "onnx.Transpose"(%o) {perm = [0, 2, 1, 3]} : (tensor<1x4x32x64xf16>) -> tensor<1x32x4x64xf16>
    return %ot : tensor<1x32x4x64xf16>
  }

  // CHECK-LABEL: func.func @probs_reused
  // CHECK-NOT: hip.multi_head_attention
  // CHECK: hip.miopen.softmax
  func.func @probs_reused(%q: tensor<1x32x256xf16>, %k: tensor<1x32x256xf16>,
                          %v: tensor<1x32x256xf16>)
      -> (tensor<1x32x4x64xf16>, tensor<1x4x32x32xf16>) {
    %split = "onnx.Constant"() {value = dense<[1, 32, 4, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
    %q4 = "onnx.Reshape"(%q, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %qt = "onnx.Transpose"(%q4) {perm = [0, 2, 1, 3]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x32x64xf16>
    %k4 = "onnx.Reshape"(%k, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %kt = "onnx.Transpose"(%k4) {perm = [0, 2, 3, 1]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x64x32xf16>
    %s = "onnx.MatMul"(%qt, %kt) : (tensor<1x4x32x64xf16>, tensor<1x4x64x32xf16>) -> tensor<1x4x32x32xf16>
    %p = "onnx.Softmax"(%s) {axis = -1 : si64} : (tensor<1x4x32x32xf16>) -> tensor<1x4x32x32xf16>
    %v4 = "onnx.Reshape"(%v, %split) : (tensor<1x32x256xf16>, tensor<4xi64>) -> tensor<1x32x4x64xf16>
    %vt = "onnx.Transpose"(%v4) {perm = [0, 2, 1, 3]} : (tensor<1x32x4x64xf16>) -> tensor<1x4x32x64xf16>
    %o = "onnx.MatMul"(%p, %vt) : (tensor<1x4x32x32xf16>, tensor<1x4x32x64xf16>) -> tensor<1x4x32x64xf16>
    %ot = "onnx.Transpose"(%o) {perm = [0, 2, 1, 3]} : (tensor<1x4x32x64xf16>) -> tensor<1x32x4x64xf16>
    return %ot, %p : tensor<1x32x4x64xf16>, tensor<1x4x32x32xf16>
  }
}
