// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// E2E pipeline check for the bidirectional LSTM that Kokoro's duration
// predictor uses (hidden=256, batch=1, seq=8, fp16).  Compiles all the
// way to LLVM IR and confirms the runtime call landed.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_miopenRNNForwardInference
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.LSTM

module {
  func.func @main_graph(%X: tensor<8x1x32xf16> {onnx.name = "input_0"})
      -> (tensor<8x2x1x256xf16> {onnx.name = "output_0"}) {
    %W  = "onnx.Constant"() {value = dense<0.01> : tensor<2x1024x32xf16>}  : () -> tensor<2x1024x32xf16>
    %R  = "onnx.Constant"() {value = dense<0.01> : tensor<2x1024x256xf16>} : () -> tensor<2x1024x256xf16>
    %B  = "onnx.Constant"() {value = dense<0.0>  : tensor<2x2048xf16>}     : () -> tensor<2x2048xf16>
    %SL = "onnx.Constant"() {value = dense<8>    : tensor<1xi32>}          : () -> tensor<1xi32>
    %iH = "onnx.Constant"() {value = dense<0.0>  : tensor<2x1x256xf16>}    : () -> tensor<2x1x256xf16>
    %iC = "onnx.Constant"() {value = dense<0.0>  : tensor<2x1x256xf16>}    : () -> tensor<2x1x256xf16>
    %y, %y_h, %y_c = "onnx.LSTM"(%X, %W, %R, %B, %SL, %iH, %iC) {
      direction = "bidirectional",
      hidden_size = 256 : si64,
      layout = 0 : si64,
      onnx_node_name = "LSTM_0"
    } : (tensor<8x1x32xf16>, tensor<2x1024x32xf16>, tensor<2x1024x256xf16>,
         tensor<2x2048xf16>, tensor<1xi32>,
         tensor<2x1x256xf16>, tensor<2x1x256xf16>)
      -> (tensor<8x2x1x256xf16>, tensor<2x1x256xf16>, tensor<2x1x256xf16>)
    "onnx.Return"(%y) : (tensor<8x2x1x256xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
