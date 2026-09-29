// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ArgMax is lowered to hip.arg_max.
//
// The text-encoder shape is rank-2 i32 with axis = -1 and keepdims = 0,
// which drops the last dim and leaves a rank-1 i64 index tensor.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128xi64>) -> tensor<1x128xi64> {
    return %arg0 : tensor<1x128xi64>
  }

  // Text-encoder ArgMax: dynamic rank-2 i32, last axis, keepdims = 0.
  func.func @test_argmax_text_encoder(%data: tensor<?x?xi32>) -> tensor<?xi64> {
    // CHECK-LABEL: func.func @test_argmax_text_encoder
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<?x?xi32>) -> tensor<?xi64>
    %output = "onnx.ArgMax"(%data) {axis = -1 : si64, keepdims = 0 : si64, select_last_index = 0 : si64} : (tensor<?x?xi32>) -> tensor<?xi64>
    // CHECK: %[[DIM:.*]] = tensor.dim %[[DATA]], %{{.*}} : tensor<?x?xi32>
    // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM]]) : tensor<?xi64>
    // CHECK: hip.arg_max(%[[CTX]]) ins(%[[DATA]] : tensor<?x?xi32>) outs(%[[INIT]] : tensor<?xi64>) {axis = 1 : i64, keepdims = 0 : i64}
    // CHECK-NOT: onnx.ArgMax
    return %output : tensor<?xi64>
  }

  // keepdims = 1 keeps the reduced axis as size 1.
  func.func @test_argmax_keepdims(%data: tensor<4x8xi32>) -> tensor<4x1xi64> {
    // CHECK-LABEL: func.func @test_argmax_keepdims
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<4x8xi32>) -> tensor<4x1xi64>
    %output = "onnx.ArgMax"(%data) {axis = 1 : si64, keepdims = 1 : si64, select_last_index = 0 : si64} : (tensor<4x8xi32>) -> tensor<4x1xi64>
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x1xi64>
    // CHECK: hip.arg_max(%[[CTX]]) ins(%[[DATA]] : tensor<4x8xi32>) outs(%[[INIT]] : tensor<4x1xi64>) {axis = 1 : i64}
    // CHECK-NOT: onnx.ArgMax
    return %output : tensor<4x1xi64>
  }

  // Reducing axis 0 drops the leading dim, so the remaining dynamic dim is
  // input dim 1, not dim 0.
  func.func @test_argmax_axis0(%data: tensor<?x?xi32>) -> tensor<?xi64> {
    // CHECK-LABEL: func.func @test_argmax_axis0
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<?x?xi32>) -> tensor<?xi64>
    %output = "onnx.ArgMax"(%data) {axis = 0 : si64, keepdims = 0 : si64, select_last_index = 1 : si64} : (tensor<?x?xi32>) -> tensor<?xi64>
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[DIM:.*]] = tensor.dim %[[DATA]], %[[C1]] : tensor<?x?xi32>
    // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM]]) : tensor<?xi64>
    // CHECK: hip.arg_max(%[[CTX]]) ins(%[[DATA]] : tensor<?x?xi32>) outs(%[[INIT]] : tensor<?xi64>) {keepdims = 0 : i64, select_last_index = 1 : i64}
    // CHECK-NOT: onnx.ArgMax
    return %output : tensor<?xi64>
  }
}
