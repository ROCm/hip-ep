// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ConvInteger accepts a runtime-sized batch when channels, spatial dimensions,
// and weights are static. The dynamic extent reaches wrap_conv through the
// input memref descriptor.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK-NOT: onnx.ConvInteger
// CHECK: llvm.func @wrap_conv
// CHECK: llvm.func @inference_compute
module {
  func.func @main_graph(
      %arg0: tensor<?x3x16x16xui8> {onnx.name = "input"},
      %arg1: tensor<2x3x16x16xi8> {onnx.name = "weight"})
      -> (tensor<?x2x1x1xi32> {onnx.name = "output"}) {
    %conv = "onnx.ConvInteger"(%arg0, %arg1) {
      kernel_shape = [16, 16],
      strides = [16, 16],
      dilations = [1, 1],
      group = 1 : i64,
      auto_pad = "VALID"
    } : (tensor<?x3x16x16xui8>, tensor<2x3x16x16xi8>)
        -> tensor<?x2x1x1xi32>
    "onnx.Return"(%conv) : (tensor<?x2x1x1xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
