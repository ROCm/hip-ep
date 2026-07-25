// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test ConvTranspose E2E pipeline: single 2D transposed convolution with bias
// Input:   tensor<1x1x3x3xf32>   (batch=1, 1 channel, 3x3 spatial)
// Weights: tensor<1x2x3x3xf32>   (C=1 input channel, M/group=2, 3x3 kernel)
// Bias:    tensor<2xf32>
// Output:  tensor<1x2x5x5xf32>   (stride=1, no pad -> 3 + (3-1) = 5 spatial)
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.ConvTranspose -> hip.conv_transpose
// 2. bufferize / canonicalize
// 3. convert-hip-to-llvm: hip.conv_transpose -> wrap_miopenConvolutionTranspose
// 4. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenConvolutionTranspose
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.ConvTranspose
module {
  func.func @main_graph(%arg0: tensor<1x1x3x3xf32> {onnx.name = "input_0"}) -> (tensor<1x2x5x5xf32> {onnx.name = "output_0"}) {
    %weights = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<1x2x3x3xf32>} : () -> tensor<1x2x3x3xf32>
    %bias = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "onnx.ConvTranspose"(%arg0, %weights, %bias) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      output_padding = [0, 0],
      strides = [1, 1],
      onnx_node_name = "ConvTranspose_0"
    } : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, tensor<2xf32>) -> tensor<1x2x5x5xf32>
    "onnx.Return"(%0) : (tensor<1x2x5x5xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
