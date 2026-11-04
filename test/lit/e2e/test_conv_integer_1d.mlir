// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Rank-3 ConvInteger widens NCL to NC1L for hip.conv and collapses the i32
// result back to NCL before the generated output ABI sees it.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK-NOT: onnx.ConvInteger
// CHECK: llvm.func @wrap_conv
// CHECK: llvm.func @inference_compute
module {
  func.func @main_graph(
      %arg0: tensor<1x80x3000xui8> {onnx.name = "input"},
      %arg1: tensor<1280x80x3xi8> {onnx.name = "weight"})
      -> (tensor<1x1280x3000xi32> {onnx.name = "output"}) {
    %conv = "onnx.ConvInteger"(%arg0, %arg1) {
      kernel_shape = [3],
      strides = [1],
      pads = [1, 1],
      dilations = [1],
      group = 1 : i64,
      auto_pad = "NOTSET"
    } : (tensor<1x80x3000xui8>, tensor<1280x80x3xi8>) -> tensor<1x1280x3000xi32>
    "onnx.Return"(%conv) : (tensor<1x1280x3000xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
