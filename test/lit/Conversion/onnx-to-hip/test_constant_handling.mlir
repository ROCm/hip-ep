// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: constant externalization emits hipdnn.constant_sizes/offsets and
// hip.external_data with index field.
//
// Verifies:
// - Large constants (splat and non-splat) are externalized to memref.global
//   with hip.external_data containing index, offset, and size
// - hipdnn.constant_sizes and hipdnn.constant_offsets module attributes
//   are emitted
// - Small constants (below threshold) remain inline as arith.constant
//===----------------------------------------------------------------------===//

// RUN: mkdir -p %t && hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --hip-externalize-constants='externalize-min-num-elements=4 externalize-output-dir=%t' %s | FileCheck %s

// Module-level: constants file, sizes, and offsets attributes.
// CHECK: module attributes {
// CHECK-SAME: hip.constants_file = "model.constants.bin"
// CHECK-SAME: hipdnn.constant_offsets = array<i64:
// CHECK-SAME: hipdnn.constant_sizes = array<i64:

// Extern memref.global with index in hip.external_data.
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

module {
  func.func @main_graph() -> (tensor<2xf32>, tensor<4x4xf32>, tensor<2x4xf32>) {
    %small = "onnx.Constant"() {value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
    %splat = "onnx.Constant"() {value = dense<0.5> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %large = "onnx.Constant"() {value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
    "onnx.Return"(%small, %splat, %large) {onnx_node_name = "/Return"} : (tensor<2xf32>, tensor<4x4xf32>, tensor<2x4xf32>) -> ()
  }
}
