// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_lrn
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.LRN
module {
  func.func @main_graph(%arg0: tensor<1x64x56x56xf32> {onnx.name = "x"}) -> (tensor<1x64x56x56xf32> {onnx.name = "y"}) {
    %0 = "onnx.LRN"(%arg0) {alpha = 9.99999974E-5 : f32, beta = 7.500000e-01 : f32, bias = 1.000000e+00 : f32, onnx_node_name = "pool1/norm1_1", size = 5 : si64} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    "onnx.Return"(%0) : (tensor<1x64x56x56xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
