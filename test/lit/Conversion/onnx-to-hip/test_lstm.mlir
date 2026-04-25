// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify ONNX LSTM (opset 14) lowers to hip.lstm with the right
// direction enum, hidden_size, and 3 tensor.empty inits.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%X: tensor<8x1x32xf16>) -> tensor<8x2x1x256xf16> {
    %W  = "onnx.Constant"() {value = dense<0.01> : tensor<2x1024x32xf16>} : () -> tensor<2x1024x32xf16>
    %R  = "onnx.Constant"() {value = dense<0.01> : tensor<2x1024x256xf16>} : () -> tensor<2x1024x256xf16>
    %B  = "onnx.Constant"() {value = dense<0.0> : tensor<2x2048xf16>} : () -> tensor<2x2048xf16>
    %SL = "onnx.Constant"() {value = dense<8> : tensor<1xi32>} : () -> tensor<1xi32>
    %iH = "onnx.Constant"() {value = dense<0.0> : tensor<2x1x256xf16>} : () -> tensor<2x1x256xf16>
    %iC = "onnx.Constant"() {value = dense<0.0> : tensor<2x1x256xf16>} : () -> tensor<2x1x256xf16>

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %y, %y_h, %y_c = "onnx.LSTM"(%X, %W, %R, %B, %SL, %iH, %iC) {
      direction = "bidirectional",
      hidden_size = 256 : si64,
      layout = 0 : si64
    } : (tensor<8x1x32xf16>, tensor<2x1024x32xf16>, tensor<2x1024x256xf16>,
         tensor<2x2048xf16>, tensor<1xi32>,
         tensor<2x1x256xf16>, tensor<2x1x256xf16>)
      -> (tensor<8x2x1x256xf16>, tensor<2x1x256xf16>, tensor<2x1x256xf16>)

    // Three tensor.empty inits for Y, Y_h, Y_c.
    // CHECK: tensor.empty() : tensor<8x2x1x256xf16>
    // CHECK: tensor.empty() : tensor<2x1x256xf16>
    // CHECK: tensor.empty() : tensor<2x1x256xf16>
    // CHECK: hip.lstm(%[[CTX]])
    // CHECK-SAME: direction = 2 : i64
    // CHECK-SAME: hidden_size = 256 : i64

    "onnx.Return"(%y) : (tensor<8x2x1x256xf16>) -> ()
  }

  // Forward-only case to make sure the direction encoding flows through.
  func.func @lstm_forward(%X: tensor<4x1x4xf32>) -> tensor<4x1x1x8xf32> {
    %W  = "onnx.Constant"() {value = dense<0.01> : tensor<1x32x4xf32>} : () -> tensor<1x32x4xf32>
    %R  = "onnx.Constant"() {value = dense<0.01> : tensor<1x32x8xf32>} : () -> tensor<1x32x8xf32>

    // CHECK-LABEL: func.func @lstm_forward
    // CHECK-SAME: (%[[CTX1:.*]]: !hip.context,
    %y, %y_h, %y_c = "onnx.LSTM"(%X, %W, %R) {
      direction = "forward",
      hidden_size = 8 : si64,
      layout = 0 : si64
    } : (tensor<4x1x4xf32>, tensor<1x32x4xf32>, tensor<1x32x8xf32>)
      -> (tensor<4x1x1x8xf32>, tensor<1x1x8xf32>, tensor<1x1x8xf32>)
    // Without bias / initial_h / initial_c the conversion drops them from
    // the operand list -- the resulting hip.lstm has only the 3 required
    // ins.
    // CHECK: hip.lstm(%[[CTX1]])
    // CHECK-SAME: direction = 0 : i64
    // CHECK-SAME: hidden_size = 8 : i64

    "onnx.Return"(%y) : (tensor<4x1x1x8xf32>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
