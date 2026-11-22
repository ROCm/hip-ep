// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A 1x1 convolution whose reduction is one patch is otherwise a GEMM
// candidate. An i32 result must stay on hip.conv: hip.gemm rejects i32.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK-NOT: onnx.ConvInteger
// CHECK-NOT: wrap_gemm
// CHECK: llvm.func @wrap_conv
// CHECK-NOT: wrap_gemm
// CHECK: llvm.func @inference_compute
// CHECK-NOT: wrap_gemm
module {
  func.func @main_graph(
      %arg0: tensor<1x259x1x1xui8> {onnx.name = "input"},
      %arg1: tensor<1x259x1x1xui8> {onnx.name = "weight"})
      -> (tensor<1x1x1x1xi32> {onnx.name = "output"}) {
    %conv = "onnx.ConvInteger"(%arg0, %arg1) {
      kernel_shape = [1, 1],
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      group = 1 : i64,
      auto_pad = "NOTSET",
      onnx_node_name = "conv_integer"
    } : (tensor<1x259x1x1xui8>, tensor<1x259x1x1xui8>) -> tensor<1x1x1x1xi32>
    "onnx.Return"(%conv) : (tensor<1x1x1x1xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
