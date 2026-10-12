// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ImageScaler lowers to the existing elementwise add/mul path.
//
// output = scale * (input + bias), NCHW, bias broadcasts as 1xCx1x1.
// A zero bias is mul-only (TinyYOLOv2's 1/255 scale). A scale of 1 is
// add-only. Both together are an identity. A bias length that disagrees
// with a static channel count is left as onnx.ImageScaler.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32> {
    return %arg0 : tensor<1x3x8x8xf32>
  }

  // TinyYOLOv2 preprocessor: dynamic N, C=3, zero bias, scale = 1/255.
  func.func @imagescaler_scale_only(%input: tensor<?x3x416x416xf32>) -> tensor<?x3x416x416xf32> {
    %y = "onnx.ImageScaler"(%input) {bias = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32], scale = 0.00392156886 : f32} : (tensor<?x3x416x416xf32>) -> tensor<?x3x416x416xf32>
    return %y : tensor<?x3x416x416xf32>
  }

  // CHECK-LABEL: func.func @imagescaler_scale_only
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<?x3x416x416xf32>)
  // CHECK-NOT: onnx.ImageScaler
  // CHECK-NOT: hip.add
  // CHECK: hip.constant {{.*}} value = dense<{{.*}}> : tensor<f32>
  // CHECK: hip.mul(%[[CTX]]) ins(%[[INPUT]], {{.*}} : tensor<?x3x416x416xf32>, tensor<f32>)

  // Per-channel bias is added as 1xCx1x1, then scaled.
  func.func @imagescaler_bias_and_scale(%input: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32> {
    %y = "onnx.ImageScaler"(%input) {bias = [0.000000e+00 : f32, 1.000000e+00 : f32, -1.000000e+00 : f32], scale = 2.000000e+00 : f32} : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
    return %y : tensor<1x3x8x8xf32>
  }

  // CHECK-LABEL: func.func @imagescaler_bias_and_scale
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[INPUT2:.*]]: tensor<1x3x8x8xf32>)
  // CHECK-NOT: onnx.ImageScaler
  // CHECK: hip.constant {{.*}} value = dense<{{.*}}> : tensor<1x3x1x1xf32>
  // CHECK: hip.add(%[[CTX2]]) ins(%[[INPUT2]], {{.*}} : tensor<1x3x8x8xf32>, tensor<1x3x1x1xf32>)
  // CHECK: hip.constant {{.*}} value = dense<{{.*}}> : tensor<f32>
  // CHECK: hip.mul(%[[CTX2]])

  // scale defaults to 1, so a non-zero bias is add-only.
  func.func @imagescaler_bias_only(%input: tensor<2x3x4x4xf16>) -> tensor<2x3x4x4xf16> {
    %y = "onnx.ImageScaler"(%input) {bias = [1.000000e+00 : f32, 0.000000e+00 : f32, -2.000000e+00 : f32]} : (tensor<2x3x4x4xf16>) -> tensor<2x3x4x4xf16>
    return %y : tensor<2x3x4x4xf16>
  }

  // CHECK-LABEL: func.func @imagescaler_bias_only
  // CHECK-NOT: onnx.ImageScaler
  // CHECK-NOT: hip.mul
  // CHECK: hip.constant {{.*}} : tensor<1x3x1x1xf16>
  // CHECK: hip.add

  // scale == 1 and a zero bias replace the op with its input.
  func.func @imagescaler_identity(%input: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
    %y = "onnx.ImageScaler"(%input) {bias = [0.000000e+00 : f32], scale = 1.000000e+00 : f32} : (tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    return %y : tensor<1x1x2x2xf32>
  }

  // CHECK-LABEL: func.func @imagescaler_identity
  // CHECK-SAME: (%[[CTX4:.*]]: !hip.context, %[[INPUT4:.*]]: tensor<1x1x2x2xf32>)
  // CHECK-NOT: onnx.ImageScaler
  // CHECK-NOT: hip.add
  // CHECK-NOT: hip.mul
  // CHECK: return %[[INPUT4]]

  // Bias length 2 cannot broadcast onto C=3.
  func.func @imagescaler_bias_length_mismatch(%input: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32> {
    %y = "onnx.ImageScaler"(%input) {bias = [0.000000e+00 : f32, 1.000000e+00 : f32], scale = 1.000000e+00 : f32} : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
    return %y : tensor<1x3x8x8xf32>
  }

  // CHECK-LABEL: func.func @imagescaler_bias_length_mismatch
  // CHECK: onnx.ImageScaler
  // CHECK-NOT: hip.add
  // CHECK-NOT: hip.mul
}
