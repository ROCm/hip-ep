// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A static batch and a dynamic sequence keep M runtime-sized. K and N are
// static, and the rank-2 weight is one shared matrix.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK-NOT: onnx.MatMulInteger
// CHECK: llvm.func @wrap_matmul_i32
// CHECK: llvm.func @inference_compute
module {
  func.func @main_graph(
      %arg0: tensor<1x?x300xui8> {onnx.name = "input"},
      %arg1: tensor<300x4xi8> {onnx.name = "weight"})
      -> (tensor<1x?x4xi32> {onnx.name = "output"}) {
    %matmul = "onnx.MatMulInteger"(%arg0, %arg1) :
        (tensor<1x?x300xui8>, tensor<300x4xi8>) -> tensor<1x?x4xi32>
    "onnx.Return"(%matmul) : (tensor<1x?x4xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
