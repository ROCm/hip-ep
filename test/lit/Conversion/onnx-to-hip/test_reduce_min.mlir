// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ReduceMin is correctly lowered to hip.reduce_min operation.
//
// Test cases:
// 1. keepdims = 1: output shape keeps reduced dimension as size 1
// 2. keepdims = 0: output shape drops the reduced dimension entirely
// 3. Dynamic shape with keepdims = 0
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128xi64>) -> tensor<1x128xi64> {
    return %arg0 : tensor<1x128xi64>
  }

  func.func @test_reduce_min_keepdims(%data: tensor<1x128xi64>, %axes: tensor<i64>) -> tensor<1x1xi64> {
    // CHECK-LABEL: func.func @test_reduce_min_keepdims
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<1x128xi64>, %[[AXES:.*]]: tensor<i64>) -> tensor<1x1xi64>

    %output = "onnx.ReduceMin"(%data, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x128xi64>, tensor<i64>) -> tensor<1x1xi64>

    // CHECK: tensor.empty() : tensor<1x1xi64>
    // CHECK: hip.reduce_min(%[[CTX]]) ins(%[[DATA]], %[[AXES]] : tensor<1x128xi64>, tensor<i64>) outs({{.*}} : tensor<1x1xi64>)
    // CHECK-NOT: hip.alloc

    return %output : tensor<1x1xi64>
  }

  func.func @test_reduce_min_no_keepdims(%data: tensor<4x8xi64>, %axes: tensor<i64>) -> tensor<4xi64> {
    // CHECK-LABEL: func.func @test_reduce_min_no_keepdims
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<4x8xi64>, %[[AXES:.*]]: tensor<i64>) -> tensor<4xi64>

    %output = "onnx.ReduceMin"(%data, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<4x8xi64>, tensor<i64>) -> tensor<4xi64>

    // CHECK: tensor.empty() : tensor<4xi64>
    // CHECK: hip.reduce_min(%[[CTX]]) ins(%[[DATA]], %[[AXES]] : tensor<4x8xi64>, tensor<i64>) outs({{.*}} : tensor<4xi64>) {keepdims = 0 : i64}
    // CHECK-NOT: hip.alloc

    return %output : tensor<4xi64>
  }

  func.func @reduce_min_dynamic(%data: tensor<?x?x512xf32>, %axes: tensor<i64>) -> tensor<?x?xf32> {
    %output = "onnx.ReduceMin"(%data, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x?x512xf32>, tensor<i64>) -> tensor<?x?xf32>
    return %output : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @reduce_min_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<?x?x512xf32>, %[[AXES:.*]]: tensor<i64>) -> tensor<?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xf32>
  // CHECK: hip.reduce_min(%[[CTX]]) ins(%[[DATA]], %[[AXES]] : tensor<?x?x512xf32>, tensor<i64>) outs(%[[INIT]] : tensor<?x?xf32>) {keepdims = 0 : i64}
  // CHECK-NOT: hip.alloc
}
