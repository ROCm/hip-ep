// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Rank-5 NCDHW conv through the full pipeline. Depth kernel 3, pad 1, and a
// 1x1 spatial kernel keep the input shape. This overlaps, so it must stay a
// convolution (wrap_conv) rather than the patch-embed GEMM.

// CHECK: llvm.func @wrap_conv
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.Conv
// CHECK-NOT: llvm.func @wrap_miopenConvolutionForward
module {
  func.func @main_graph(%arg0: tensor<1x4x6x4x4xf16> {onnx.name = "input_0"}) -> (tensor<1x4x6x4x4xf16> {onnx.name = "output_0"}) {
    %weights = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<4x4x3x1x1xf16>} : () -> tensor<4x4x3x1x1xf16>
    %bias = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<4xf16>} : () -> tensor<4xf16>
    %0 = "onnx.Conv"(%arg0, %weights, %bias) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [3, 1, 1],
      pads = [1, 0, 0, 1, 0, 0],
      strides = [1, 1, 1],
      onnx_node_name = "Conv3d_0"
    } : (tensor<1x4x6x4x4xf16>, tensor<4x4x3x1x1xf16>, tensor<4xf16>) -> tensor<1x4x6x4x4xf16>
    "onnx.Return"(%0) : (tensor<1x4x6x4x4xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
