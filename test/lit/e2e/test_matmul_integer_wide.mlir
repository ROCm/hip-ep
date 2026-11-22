// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// K = 259 does not fit an f32 mantissa, so the matmul stays i32 and lowers
// to the integer kernel rather than hipBLASLt.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK-NOT: onnx.MatMulInteger
// CHECK-NOT: wrap_hipblasLtMatmul
// CHECK: llvm.func @wrap_matmul_i32
// CHECK-NOT: wrap_hipblasLtMatmul
// CHECK: llvm.func @inference_compute
// CHECK-NOT: wrap_hipblasLtMatmul
module {
  func.func @main_graph(
      %arg0: tensor<1x259xui8> {onnx.name = "input"},
      %arg1: tensor<259x1xui8> {onnx.name = "weight"})
      -> (tensor<1x1xi32> {onnx.name = "output"}) {
    %mm = "onnx.MatMulInteger"(%arg0, %arg1) {
      onnx_node_name = "matmul_integer"
    } : (tensor<1x259xui8>, tensor<259x1xui8>) -> tensor<1x1xi32>
    "onnx.Return"(%mm) : (tensor<1x1xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
