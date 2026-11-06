// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A constant scalar exponent that does not decompose to Mul, Sqrt, or
// Reciprocal lowers through hip.pow to wrap_pow.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_pow
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Pow
module {
  func.func @main_graph(%arg0: tensor<4x8xf32> {onnx.name = "input"}) -> (tensor<4x8xf32> {onnx.name = "output"}) {
    %exp = "onnx.Constant"() {value = dense<2.2> : tensor<f32>} : () -> tensor<f32>
    %0 = "onnx.Pow"(%arg0, %exp) {onnx_node_name = "pow_node"} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
    "onnx.Return"(%0) : (tensor<4x8xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
