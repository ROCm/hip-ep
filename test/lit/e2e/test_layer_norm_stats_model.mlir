// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_layer_normalization
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.LayerNormalization
module {
  func.func @main_graph(
      %input: tensor<2x4xf16> {onnx.name = "input"},
      %scale: tensor<4xf16> {onnx.name = "scale"})
      -> (tensor<2x4xf16> {onnx.name = "output"},
          tensor<2x1xf32> {onnx.name = "inv_std"}) {
    %result:3 = "onnx.LayerNormalization"(%input, %scale)
      {axis = -1 : si64, epsilon = 9.99999974E-6 : f32,
       stash_type = 1 : si64}
      : (tensor<2x4xf16>, tensor<4xf16>)
        -> (tensor<2x4xf16>, none, tensor<2x1xf32>)
    "onnx.Return"(%result#0, %result#2)
      : (tensor<2x4xf16>, tensor<2x1xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
