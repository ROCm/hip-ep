// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ReduceSum is correctly lowered to hip.reduce_sum operation
// in tensor-first mode.
//
// This test validates:
// - Reduction operation lowering (onnx.ReduceSum -> hip.reduce_sum)
// - keepdims = 1: output shape keeps reduced dimension as size 1
// - keepdims = 0: output shape drops the reduced dimension entirely
// - i64 element type support
// - 2D tensor reduction with axes as input operand
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
//
// Model: Llama-3.1-8B attention mask sum computation
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // keepdims = 1: reduced dim is kept as size 1
  func.func @test_reduce_sum_keepdims(%data: tensor<1x128xi64>, %axes: tensor<i64>) -> tensor<1x1xi64> {
    // After conversion: context prepended, tensors remain tensors, tensor return
    // CHECK-LABEL: func.func @test_reduce_sum_keepdims
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<1x128xi64>, %[[AXES:.*]]: tensor<i64>) -> tensor<1x1xi64>

    %output = "onnx.ReduceSum"(%data, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x128xi64>, tensor<i64>) -> tensor<1x1xi64>

    // After conversion: tensor.empty() for init, hip.reduce_sum in tensor mode
    // CHECK: tensor.empty() : tensor<1x1xi64>
    // CHECK: hip.reduce_sum(%[[CTX]], %[[DATA]], %[[AXES]], {{.*}}) {keepdims = 1 : i64}
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<1x1xi64>
  }

  // keepdims = 0: reduced dim is dropped from output shape
  func.func @test_reduce_sum_no_keepdims(%data: tensor<4x8xi64>, %axes: tensor<i64>) -> tensor<4xi64> {
    // CHECK-LABEL: func.func @test_reduce_sum_no_keepdims
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<4x8xi64>, %[[AXES:.*]]: tensor<i64>) -> tensor<4xi64>

    %output = "onnx.ReduceSum"(%data, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<4x8xi64>, tensor<i64>) -> tensor<4xi64>

    // CHECK: tensor.empty() : tensor<4xi64>
    // CHECK: hip.reduce_sum(%[[CTX]], %[[DATA]], %[[AXES]], {{.*}}) {keepdims = 0 : i64}
    // CHECK-NOT: hip.alloc

    return %output : tensor<4xi64>
  }
}
