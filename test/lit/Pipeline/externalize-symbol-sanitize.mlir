// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: symbol name sanitization during constant externalization.
//
// Verifies that onnx_node_name values containing /, :, spaces, and other
// unsafe characters are sanitized to valid MLIR identifiers, and that the
// monotonic index suffix prevents collisions.
//===----------------------------------------------------------------------===//

// RUN: mkdir -p %t && hip-mlir-opt --convert-onnx-to-hip='externalize-min-num-elements=4 externalize-output-dir=%t' %s | FileCheck %s

// The node name "/model/layers.0/Constant" should become hip_ext_constant_model_layers_0_Constant_0
// CHECK-DAG: memref.global "private" @hip_ext_constant_model_layers_0_Constant_0{{.*}}hip.external_data

// The node name "weight::bias" should become hip_ext_constant_weight_bias_1
// CHECK-DAG: memref.global "private" @hip_ext_constant_weight_bias_1{{.*}}hip.external_data

// A constant with no onnx_node_name falls back to index-only: hip_ext_constant_2
// CHECK-DAG: memref.global "private" @hip_ext_constant_2{{.*}}hip.external_data

module {
  func.func @main_graph() -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
    %a = "onnx.Constant"() {
      onnx_node_name = "/model/layers.0/Constant",
      value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
    } : () -> tensor<2x4xf32>
    %b = "onnx.Constant"() {
      onnx_node_name = "weight::bias",
      value = dense<[[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]> : tensor<2x4xf32>
    } : () -> tensor<2x4xf32>
    %c = "onnx.Constant"() {
      value = dense<[[17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]> : tensor<2x4xf32>
    } : () -> tensor<2x4xf32>
    "onnx.Return"(%a, %b, %c) {onnx_node_name = "/Return"} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> ()
  }
}
