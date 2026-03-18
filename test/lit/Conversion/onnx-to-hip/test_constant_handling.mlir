// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: constant externalization emits hipdnn.constant_sizes/offsets and
// hip.external_data with index field.
//
// Verifies:
// - Large non-splat constants are externalized to memref.global with
//   hip.external_data containing index, offset, and size
// - hipdnn.constant_sizes and hipdnn.constant_offsets module attributes
//   are emitted
// - Small and splat constants remain inline as arith.constant
//===----------------------------------------------------------------------===//

// RUN: mkdir -p %t && hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip='externalize-min-num-elements=4 externalize-output-dir=%t' %s | FileCheck %s

// Module-level: constants file, sizes, and offsets attributes.
// CHECK: module attributes {
// CHECK-SAME: hip.constants_file = "constants.bin"
// CHECK-SAME: hipdnn.constant_offsets = array<i64:
// CHECK-SAME: hipdnn.constant_sizes = array<i64:

// Extern memref.global with index in hip.external_data.
// CHECK: memref.global "private" @hip_ext_constant_0 : memref<2x4xf32>
// CHECK-SAME: hip.external_data = {index = 0 : i64, offset = 0 : i64, size = 32 : i64}

// CHECK-LABEL: func.func @main_graph
//   Small constant stays inline.
// CHECK-DAG:   arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
//   Splat constant stays inline.
// CHECK-DAG:   arith.constant dense<5.000000e-01> : tensor<4x4xf32>
//   Large non-splat loaded from extern global.
// CHECK:       memref.get_global @hip_ext_constant_0 : memref<2x4xf32>
// CHECK-NEXT:  bufferization.to_tensor {{.*}} restrict

module {
  func.func @main_graph() -> (tensor<2xf32>, tensor<4x4xf32>, tensor<2x4xf32>) {
    %small = "onnx.Constant"() {value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
    %splat = "onnx.Constant"() {value = dense<0.5> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %large = "onnx.Constant"() {value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
    "onnx.Return"(%small, %splat, %large) {onnx_node_name = "/Return"} : (tensor<2xf32>, tensor<4x4xf32>, tensor<2x4xf32>) -> ()
  }
}
