// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// DynamicQuantizeLinear followed by MatMulInteger, the logits head of a
// quantized classifier. Both ops decompose to HIP ops that bufferize;
// hip.matmul then lowers to hipBLASLt.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.DynamicQuantizeLinear
// CHECK-NOT: onnx.MatMulInteger
// CHECK-DAG: llvm.func @wrap_hipblasLtMatmul
// CHECK-DAG: llvm.func @wrap_cast
// CHECK-DAG: llvm.func @wrap_elementwise_sub
// CHECK-DAG: llvm.func @wrap_quantize_linear
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(
      %arg0: tensor<1x4xf32> {onnx.name = "input"},
      %arg1: tensor<4x2xui8> {onnx.name = "weight"},
      %arg2: tensor<2xui8> {onnx.name = "w_zp"})
      -> (tensor<1x2xi32> {onnx.name = "output"}) {
    %y, %scale, %zp = "onnx.DynamicQuantizeLinear"(%arg0) {
      onnx_node_name = "dynamic_quantize_linear"
    } : (tensor<1x4xf32>) -> (tensor<1x4xui8>, tensor<f32>, tensor<ui8>)
    %mm = "onnx.MatMulInteger"(%y, %arg1, %zp, %arg2) {
      onnx_node_name = "matmul_integer"
    } : (tensor<1x4xui8>, tensor<4x2xui8>, tensor<ui8>, tensor<2xui8>)
        -> tensor<1x2xi32>
    "onnx.Return"(%mm) : (tensor<1x2xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
