// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: module metadata with multiple outputs.
//
// Verifies metadata generation for a @main_graph with two outputs of
// different shapes and element types.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1 : i64
// CHECK-SAME: hipdnn.input_element_sizes = array<i64: 4>
// CHECK-SAME: hipdnn.input_shapes = [array<i64: 2, 4>]
// CHECK-SAME: hipdnn.output_count = 2 : i64
// CHECK-SAME: hipdnn.output_element_sizes = array<i64: 4, 2>
// CHECK-SAME: hipdnn.output_shapes = [array<i64: 2, 4>, array<i64: 4>]

module {
  func.func @main_graph(%arg0: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<4xf16>) {
    %cst = "onnx.Constant"() {value = dense<1.0> : tensor<4xf16>} : () -> tensor<4xf16>
    "onnx.Return"(%arg0, %cst) {onnx_node_name = "/Return"} : (tensor<2x4xf32>, tensor<4xf16>) -> ()
  }
}
