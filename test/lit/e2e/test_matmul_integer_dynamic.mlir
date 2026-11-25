// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Dynamic leading dimensions, including a runtime sequence in M, reach the
// integer matmul runtime wrapper through A's memref descriptor. The rank-2 B
// is one shared matrix.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK-NOT: onnx.MatMulInteger
// CHECK: llvm.func @wrap_matmul_i32
// CHECK: llvm.func @inference_compute
module {
  func.func @main_graph(
      %arg0: tensor<?x2x300xui8> {onnx.name = "input"},
      %arg1: tensor<300x4xi8> {onnx.name = "weight"})
      -> (tensor<?x2x4xi32> {onnx.name = "output"}) {
    %matmul = "onnx.MatMulInteger"(%arg0, %arg1) :
        (tensor<?x2x300xui8>, tensor<300x4xi8>) -> tensor<?x2x4xi32>
    "onnx.Return"(%matmul) : (tensor<?x2x4xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
