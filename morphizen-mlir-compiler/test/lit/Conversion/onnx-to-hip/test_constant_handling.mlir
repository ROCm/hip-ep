// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Constant operations are correctly lowered to hip.get_constant
// operations with index-based lookup.
//
// This test validates:
// - onnx.Constant → hip.get_constant conversion
// - Constant discovery and global index assignment
// - Index-based retrieval instead of value embedding
// - Type conversion: tensor<...> → memref<..., 1> (GPU address space)
// - Multiple constant handling with unique indices
//
// Implementation detail:
// - Constants are discovered in Phase 1 and assigned global indices
// - Converted to hip.get_constant(context, index) in Phase 2
// - Actual constant data is pre-uploaded to GPU at runtime
//
// Expected: hip.get_constant with arith.constant index, not embedded values
// ============================================================================

// RUN: morphizen-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @constant_test() -> tensor<3xf32> {
    // CHECK-LABEL: func.func @constant_test
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[OUTPUT_ARG:.*]]: memref<3xf32, 1>) -> i32

    // ONNX Constant operation with dense elements
    // Gets converted to hip.get_constant with an index lookup
    %const = "onnx.Constant"() {
      value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    } : () -> tensor<3xf32>

    // CHECK: %[[INDEX:.*]] = arith.constant {{[0-9]+}} : i64
    // CHECK: %[[CONST:.*]] = hip.get_constant(%[[CTX]], %[[INDEX]]) : memref<3xf32, 1>

    return %const : tensor<3xf32>
  }

  func.func @constant_matrix() -> tensor<2x2xf32> {
    // CHECK-LABEL: func.func @constant_matrix
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[OUTPUT_ARG:.*]]: memref<2x2xf32, 1>) -> i32

    // 2D constant tensor
    // Constants are discovered and assigned global indices, then retrieved by index
    %matrix = "onnx.Constant"() {
      value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    } : () -> tensor<2x2xf32>

    // CHECK: %[[INDEX:.*]] = arith.constant {{[0-9]+}} : i64
    // CHECK: %[[MATRIX:.*]] = hip.get_constant(%[[CTX]], %[[INDEX]]) : memref<2x2xf32, 1>

    return %matrix : tensor<2x2xf32>
  }

  func.func @multiple_constants() -> (tensor<4xf32>, tensor<4xf32>) {
    // CHECK-LABEL: func.func @multiple_constants
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[OUTPUT_ARG1:.*]]: memref<4xf32, 1>, %[[OUTPUT_ARG2:.*]]: memref<4xf32, 1>) -> i32

    // Test that multiple constants get different indices
    %const1 = "onnx.Constant"() {
      value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    } : () -> tensor<4xf32>

    %const2 = "onnx.Constant"() {
      value = dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
    } : () -> tensor<4xf32>

    // Each constant should get a unique index
    // CHECK: arith.constant {{[0-9]+}} : i64
    // CHECK: hip.get_constant
    // CHECK: arith.constant {{[0-9]+}} : i64
    // CHECK: hip.get_constant

    return %const1, %const2 : tensor<4xf32>, tensor<4xf32>
  }
}
