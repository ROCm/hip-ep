// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Conv E2E pipeline: single 2D convolution with bias
// Input:   tensor<1x3x8x8xf32>   (batch=1, 3 channels, 8x8 spatial)
// Weights: tensor<16x3x3x3xf32>  (16 filters, 3x3 kernel)
// Bias:    tensor<16xf32>
// Output:  tensor<1x16x8x8xf32>  (stride=1, padding=1 preserves size)
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Conv -> hip.conv
// 2. bufferize / canonicalize
// 3. convert-hip-to-llvm: hip.conv -> wrap_miopenConvolutionForward
// 4. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenConvolutionForward
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Conv
module {
  func.func @main_graph(%arg0: tensor<1x3x8x8xf32> {onnx.name = "input_0"}) -> (tensor<1x16x8x8xf32> {onnx.name = "output_0"}) {
    %weights = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<16x3x3x3xf32>} : () -> tensor<16x3x3x3xf32>
    %bias = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %0 = "onnx.Conv"(%arg0, %weights, %bias) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1],
      onnx_node_name = "Conv_0"
    } : (tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x8x8xf32>
    "onnx.Return"(%0) : (tensor<1x16x8x8xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
