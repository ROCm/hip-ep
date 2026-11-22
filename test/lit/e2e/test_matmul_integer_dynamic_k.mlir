// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// An uninferred K on A still reaches the integer matmul wrapper. B's K and N
// are static, and the accumulator stays i32.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK-NOT: onnx.MatMulInteger
// CHECK: llvm.func @wrap_matmul_i32
// CHECK: llvm.func @inference_compute
module {
  func.func @main_graph(
      %arg0: tensor<?x?x?xui8> {onnx.name = "input"},
      %arg1: tensor<512x512xi8> {onnx.name = "weight"})
      -> (tensor<?x?x512xi32> {onnx.name = "output"}) {
    %matmul = "onnx.MatMulInteger"(%arg0, %arg1) :
        (tensor<?x?x?xui8>, tensor<512x512xi8>) -> tensor<?x?x512xi32>
    "onnx.Return"(%matmul) : (tensor<?x?x512xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
