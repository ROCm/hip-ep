// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: constant externalization in convert-onnx-to-hip.
//
// Verifies that --externalize-min-num-elements selectively externalizes
// constants:
//   - Large constants (splat and non-splat) -> memref.global with hip.external_data
//   - Small constants (below threshold) -> arith.constant (inline)
//
// Also verifies module-level metadata:
//   - hip.constants_file attribute
//   - hipdnn.constant_sizes and hipdnn.constant_offsets arrays
//   - hip.external_data includes index field
//
// All three constants are returned so they survive DCE in the greedy
// pattern rewrite driver that runs as part of convertComputeOps.
//===----------------------------------------------------------------------===//

// RUN: mkdir -p %t && hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip='externalize-min-num-elements=4 externalize-output-dir=%t' %s | FileCheck %s

// Module-level: constants file attribute, sizes, and offsets.
// CHECK: module attributes {
// CHECK-SAME: hip.constants_file = "model.constants.bin"
// CHECK-SAME: hipdnn.constant_offsets = array<i64:
// CHECK-SAME: hipdnn.constant_sizes = array<i64:

// Splat constant (4x4, 16 elements >= threshold) externalized as index 0.
// CHECK-DAG: memref.global "private" @hip_ext_constant_0 : memref<4x4xf32>
// Non-splat constant (2x4, 8 elements >= threshold) externalized as index 1.
// CHECK-DAG: memref.global "private" @hip_ext_constant_1 : memref<2x4xf32>

// CHECK-LABEL: func.func @main_graph
//   Small constant stays inline (2 elements < threshold 4).
// CHECK-DAG:   arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
//   Splat and non-splat loaded from extern globals.
// CHECK-DAG:   memref.get_global @hip_ext_constant_0 : memref<4x4xf32>
// CHECK-DAG:   memref.get_global @hip_ext_constant_1 : memref<2x4xf32>
// CHECK:       return

module {
  func.func @main_graph() -> (tensor<2xf32>, tensor<4x4xf32>, tensor<2x4xf32>) {
    // Small: 2 elements (below threshold of 4).
    %small = "onnx.Constant"() {value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
    // Splat: 16 elements (above threshold) -- externalized with expanded data.
    %splat = "onnx.Constant"() {value = dense<0.5> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    // Large non-splat: 8 elements (above threshold of 4).
    %large = "onnx.Constant"() {value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
    "onnx.Return"(%small, %splat, %large) {onnx_node_name = "/Return"} : (tensor<2xf32>, tensor<4x4xf32>, tensor<2x4xf32>) -> ()
  }
}
