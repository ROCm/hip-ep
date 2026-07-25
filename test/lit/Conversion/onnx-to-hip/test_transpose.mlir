// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata in convert-onnx-to-hip.
  func.func @main_graph(%arg0: tensor<i32>) -> tensor<i32> {
    return %arg0 : tensor<i32>
  }

  // Explicit perm permuting all three dims: [1, 2, 3] -> [3, 1, 2]
  func.func @test_transpose_3d_perm(%arg0: tensor<1x2x3xf32>) -> tensor<3x1x2xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [2, 0, 1]} : (tensor<1x2x3xf32>) -> tensor<3x1x2xf32>
    return %0 : tensor<3x1x2xf32>
  }
  // CHECK-LABEL: func.func @test_transpose_3d_perm
  // CHECK-SAME: !hip.context
  // CHECK: tensor.empty
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [2, 0, 1]

  // Default perm (omitted attribute) -> reverse: [1, 2, 3] -> [3, 2, 1]
  func.func @test_transpose_default_perm(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
    %0 = "onnx.Transpose"(%arg0) : (tensor<1x2x3xf32>) -> tensor<3x2x1xf32>
    return %0 : tensor<3x2x1xf32>
  }
  // CHECK-LABEL: func.func @test_transpose_default_perm
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [2, 1, 0]

  // 2-D matrix transpose: [M, N] -> [N, M]
  func.func @test_transpose_2d(%arg0: tensor<3x5xf16>) -> tensor<5x3xf16> {
    %0 = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<3x5xf16>) -> tensor<5x3xf16>
    return %0 : tensor<5x3xf16>
  }
  // CHECK-LABEL: func.func @test_transpose_2d
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [1, 0]

  // 4-D transpose (e.g. NHWC -> NCHW)
  func.func @test_transpose_4d_nhwc_to_nchw(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x3x224x224xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<1x224x224x3xf32>) -> tensor<1x3x224x224xf32>
    return %0 : tensor<1x3x224x224xf32>
  }
  // CHECK-LABEL: func.func @test_transpose_4d_nhwc_to_nchw
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [0, 3, 1, 2]

  // Identity perm is still a valid Transpose op (no-op semantics, but lowering
  // must not crash).
  func.func @test_transpose_identity(%arg0: tensor<2x3x4xi32>) -> tensor<2x3x4xi32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 1, 2]} : (tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
    return %0 : tensor<2x3x4xi32>
  }
  // CHECK-LABEL: func.func @test_transpose_identity
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [0, 1, 2]

  // Dynamic shape: dim 0 is dynamic, must materialize a tensor.dim for the
  // matching output dim through the DPS init.
  func.func @test_transpose_dynamic(%arg0: tensor<?x4xf32>) -> tensor<4x?xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<?x4xf32>) -> tensor<4x?xf32>
    return %0 : tensor<4x?xf32>
  }
  // CHECK-LABEL: func.func @test_transpose_dynamic
  // CHECK: tensor.dim
  // CHECK: tensor.empty
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [1, 0]
}
