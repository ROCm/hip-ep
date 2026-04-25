// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test ConvTranspose E2E full pipeline (1-D-via-NCHW shape).
//
// Models a stride-2 1-D upsampling block of the kind Kokoro uses for
// iSTFT-style decoders.  Input/output spatial layout is NCHW with H=1,
// kernel_shape=[1,4], strides=[1,2], pads=[0,1,0,1] (one-pixel of W on
// each side), dilations=[1,1], output_padding=[0,0], group=1.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip:  onnx.ConvTranspose -> hip.conv_transpose
// 2. bufferize / canonicalize
// 3. convert-hip-to-llvm:  hip.conv_transpose ->
//                          wrap_miopenConvolutionBackwardData
// 4. generate-interface:   inference_init / compute / cleanup / metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenConvolutionBackwardData
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.ConvTranspose
module {
  func.func @main_graph(%arg0: tensor<1x64x1x100xf16> {onnx.name = "input_0"})
      -> (tensor<1x32x1x200xf16> {onnx.name = "output_0"}) {
    %weights = "onnx.Constant"() {
      value = dense<5.000000e-01> : tensor<64x32x1x4xf16>
    } : () -> tensor<64x32x1x4xf16>
    %bias = "onnx.Constant"() {
      value = dense<0.000000e+00> : tensor<32xf16>
    } : () -> tensor<32xf16>
    %0 = "onnx.ConvTranspose"(%arg0, %weights, %bias) {
      kernel_shape = [1, 4],
      strides = [1, 2],
      pads = [0, 1, 0, 1],
      dilations = [1, 1],
      output_padding = [0, 0],
      group = 1 : si64,
      onnx_node_name = "ConvTranspose_0"
    } : (tensor<1x64x1x100xf16>, tensor<64x32x1x4xf16>, tensor<32xf16>)
        -> tensor<1x32x1x200xf16>
    "onnx.Return"(%0) : (tensor<1x32x1x200xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
