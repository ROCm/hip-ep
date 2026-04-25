// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify onnx.STFT lowers to hip.stft, with frame_step / frame_length pulled
// from the constant operands and onesided propagated as an attribute.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }

  // ---- 2-D signal, explicit window + frame_length, onesided=1 ----
  func.func @test_stft_2d_window(%signal: tensor<1x320xf32>)
      -> tensor<1x76x11x2xf32> {
    %step = "onnx.Constant"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
    %wlen = "onnx.Constant"() {value = dense<20> : tensor<i64>} : () -> tensor<i64>
    %window = "onnx.Constant"() {
      value = dense<[0.0, 0.0244717, 0.0954915, 0.2061074, 0.3454915, 0.5,
                     0.6545085, 0.7938926, 0.9045085, 0.9755283, 1.0,
                     0.9755283, 0.9045085, 0.7938926, 0.6545085, 0.5,
                     0.3454915, 0.2061074, 0.0954915, 0.0244717]>
        : tensor<20xf32>} : () -> tensor<20xf32>
    %0 = "onnx.STFT"(%signal, %step, %window, %wlen) {onesided = 1 : si64}
        : (tensor<1x320xf32>, tensor<i64>, tensor<20xf32>, tensor<i64>)
            -> tensor<1x76x11x2xf32>
    return %0 : tensor<1x76x11x2xf32>
  }
  // CHECK-LABEL: func.func @test_stft_2d_window
  // CHECK: hip.stft(%{{.*}})
  // CHECK-SAME: frame_length = 20
  // CHECK-SAME: frame_step = 4

  // ---- 3-D (batch, signal, 1) signal: trailing unit dim is collapsed ----
  func.func @test_stft_3d_collapse(%signal: tensor<1x320x1xf32>)
      -> tensor<1x76x11x2xf32> {
    %step = "onnx.Constant"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
    %wlen = "onnx.Constant"() {value = dense<20> : tensor<i64>} : () -> tensor<i64>
    %window = "onnx.Constant"() {
      value = dense<1.0> : tensor<20xf32>} : () -> tensor<20xf32>
    %0 = "onnx.STFT"(%signal, %step, %window, %wlen) {onesided = 1 : si64}
        : (tensor<1x320x1xf32>, tensor<i64>, tensor<20xf32>, tensor<i64>)
            -> tensor<1x76x11x2xf32>
    return %0 : tensor<1x76x11x2xf32>
  }
  // CHECK-LABEL: func.func @test_stft_3d_collapse
  // CHECK: tensor.collapse_shape
  // CHECK: hip.stft(%{{.*}})

  // ---- frame_length absent -- inferred from window shape ----
  func.func @test_stft_no_frame_length(%signal: tensor<1x320xf32>)
      -> tensor<1x76x11x2xf32> {
    %step = "onnx.Constant"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
    %window = "onnx.Constant"() {
      value = dense<1.0> : tensor<20xf32>} : () -> tensor<20xf32>
    %0 = "onnx.STFT"(%signal, %step, %window) {onesided = 1 : si64}
        : (tensor<1x320xf32>, tensor<i64>, tensor<20xf32>)
            -> tensor<1x76x11x2xf32>
    return %0 : tensor<1x76x11x2xf32>
  }
  // CHECK-LABEL: func.func @test_stft_no_frame_length
  // CHECK: hip.stft(%{{.*}})
  // CHECK-SAME: frame_length = 20

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
