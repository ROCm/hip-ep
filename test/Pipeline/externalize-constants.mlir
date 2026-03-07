// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: constant externalization in convert-onnx-to-hip.
//
// Verifies that --externalize-min-num-elements selectively externalizes
// constants:
//   - Large non-splat constants -> memref.global with hip.external_data
//   - Small constants (below threshold) -> arith.constant (inline)
//   - Splat constants (any size) -> arith.constant (inline)
//
// All three constants are returned so they survive DCE in the greedy
// pattern rewrite driver that runs as part of convertComputeOps.
//===----------------------------------------------------------------------===//

// RUN: mkdir -p %t && %hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip='externalize-min-num-elements=4 externalize-output-dir=%t' %s | %FileCheck %s

// Module-level: constants file attribute and extern memref.global.
// CHECK: module attributes {{{.*}}hip.constants_file = "model.constants.bin"
// CHECK: memref.global "private" @hip_ext_constant_0 : memref<2x4xf32>
// CHECK-SAME: hip.external_data = {offset = 0 : i64, size = 32 : i64}

// CHECK-LABEL: func.func @test_externalize
//   Small constant stays inline (2 elements < threshold 4).
// CHECK-DAG:   arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
//   Splat constant stays inline despite 16 elements >= threshold.
// CHECK-DAG:   arith.constant dense<5.000000e-01> : tensor<4x4xf32>
//   Large non-splat constant loaded from extern global.
// CHECK:       memref.get_global @hip_ext_constant_0 : memref<2x4xf32>
// CHECK-NEXT:  bufferization.to_tensor {{.*}} restrict
// CHECK:       return

module {
  func.func @test_externalize() -> (tensor<2xf32>, tensor<4x4xf32>, tensor<2x4xf32>) {
    // Small: 2 elements (below threshold of 4).
    %small = "onnx.Constant"() {value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
    // Splat: 16 elements (above threshold) but splat -- stays inline.
    %splat = "onnx.Constant"() {value = dense<0.5> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    // Large non-splat: 8 elements (above threshold of 4).
    %large = "onnx.Constant"() {value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
    "onnx.Return"(%small, %splat, %large) {onnx_node_name = "/Return"} : (tensor<2xf32>, tensor<4x4xf32>, tensor<2x4xf32>) -> ()
  }
}
