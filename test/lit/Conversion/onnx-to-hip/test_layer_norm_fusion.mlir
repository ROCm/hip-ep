// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A primitive LayerNorm chain over the last axis folds into one
// hip.layer_norm (fp32 accumulation), both with Pow(d, 2) and with the
// Mul(d, d) form; a chain over a non-last axis is left alone.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @ln_pow(%x: tensor<1x16x768xf16>, %g: tensor<768xf16>, %b: tensor<768xf16>) -> tensor<1x16x768xf16> {
    %axes = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %two = "onnx.Constant"() {value = dense<2.0> : tensor<f16>} : () -> tensor<f16>
    %eps = "onnx.Constant"() {value = dense<9.99999997E-7> : tensor<f16>} : () -> tensor<f16>
    %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64} : (tensor<1x16x768xf16>, tensor<1xi64>) -> tensor<1x16x1xf16>
    %d = "onnx.Sub"(%x, %mean) : (tensor<1x16x768xf16>, tensor<1x16x1xf16>) -> tensor<1x16x768xf16>
    %sq = "onnx.Pow"(%d, %two) : (tensor<1x16x768xf16>, tensor<f16>) -> tensor<1x16x768xf16>
    %var = "onnx.ReduceMean"(%sq, %axes) {keepdims = 1 : si64} : (tensor<1x16x768xf16>, tensor<1xi64>) -> tensor<1x16x1xf16>
    %ve = "onnx.Add"(%var, %eps) : (tensor<1x16x1xf16>, tensor<f16>) -> tensor<1x16x1xf16>
    %s = "onnx.Sqrt"(%ve) : (tensor<1x16x1xf16>) -> tensor<1x16x1xf16>
    %n = "onnx.Div"(%d, %s) : (tensor<1x16x768xf16>, tensor<1x16x1xf16>) -> tensor<1x16x768xf16>
    %m = "onnx.Mul"(%n, %g) : (tensor<1x16x768xf16>, tensor<768xf16>) -> tensor<1x16x768xf16>
    %y = "onnx.Add"(%m, %b) : (tensor<1x16x768xf16>, tensor<768xf16>) -> tensor<1x16x768xf16>
    return %y : tensor<1x16x768xf16>
  }

  // CHECK-LABEL: func.func @ln_pow
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x16x768xf16>, %[[G:.*]]: tensor<768xf16>, %[[B:.*]]: tensor<768xf16>)
  // CHECK-NOT: hip.reduce_mean
  // CHECK: %[[Y:.*]] = hip.layer_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[G]], %[[B]] :
  // CHECK-NOT: hip.sqrt
  // CHECK: return %[[Y]]

  func.func @ln_mul_no_bias(%x: tensor<2x8xf32>, %g: tensor<8xf32>) -> tensor<2x8xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %ve = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %y = "onnx.Mul"(%g, %n) : (tensor<8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @ln_mul_no_bias
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x8xf32>, %[[G:.*]]: tensor<8xf32>)
  // CHECK-NOT: hip.reduce_mean
  // CHECK: %[[Y:.*]] = hip.layer_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[G]] :
  // CHECK: return %[[Y]]

  func.func @not_last_axis(%x: tensor<8x2xf32>, %g: tensor<2xf32>) -> tensor<8x2xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [0], keepdims = 1 : si64} : (tensor<8x2xf32>) -> tensor<1x2xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<8x2xf32>, tensor<1x2xf32>) -> tensor<8x2xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [0], keepdims = 1 : si64} : (tensor<8x2xf32>) -> tensor<1x2xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<1x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<8x2xf32>, tensor<1x2xf32>) -> tensor<8x2xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<8x2xf32>, tensor<2xf32>) -> tensor<8x2xf32>
    return %y : tensor<8x2xf32>
  }

  // CHECK-LABEL: func.func @not_last_axis
  // CHECK-NOT: hip.layer_norm
  // CHECK: hip.reduce_mean
  // CHECK: return
}
