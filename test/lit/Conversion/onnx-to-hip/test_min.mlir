// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Min is correctly lowered to hip.min, reusing the MIOpen
// miopenOpTensor path (tensor_op = kTensorOpMin = 2).
//
// Test cases:
// 1. 2-input min (basic binary case)
// 2. 3-input min (variadic → pairwise chaining)
// 3. 1-input min (identity pass-through)
// 4. Dynamic shape
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    return %arg0 : tensor<3xf32>
  }

  // --- Case 1: 2-input min ---
  func.func @min_2_inputs(%a: tensor<3xf32>, %b: tensor<3xf32>) -> tensor<3xf32> {
    %result = "onnx.Min"(%a, %b) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %result : tensor<3xf32>
  }

  // CHECK-LABEL: func.func @min_2_inputs
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<3xf32>, %[[B:.*]]: tensor<3xf32>)
  // CHECK-NOT: onnx.Min
  // CHECK: tensor.empty() : tensor<3xf32>
  // CHECK: hip.min(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<3xf32>, tensor<3xf32>) outs({{.*}} : tensor<3xf32>)

  // --- Case 2: 3-input min (variadic chaining) ---
  func.func @min_3_inputs(%a: tensor<4xf16>, %b: tensor<4xf16>, %c: tensor<4xf16>) -> tensor<4xf16> {
    %result = "onnx.Min"(%a, %b, %c) : (tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) -> tensor<4xf16>
    return %result : tensor<4xf16>
  }

  // CHECK-LABEL: func.func @min_3_inputs
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<4xf16>, %[[B:.*]]: tensor<4xf16>, %[[C:.*]]: tensor<4xf16>)
  // CHECK-NOT: onnx.Min
  // CHECK: hip.min(%[[CTX]])
  // CHECK: hip.min(%[[CTX]])

  // --- Case 3: 1-input min (identity) ---
  func.func @min_1_input(%a: tensor<3xf32>) -> tensor<3xf32> {
    %result = "onnx.Min"(%a) : (tensor<3xf32>) -> tensor<3xf32>
    return %result : tensor<3xf32>
  }

  // CHECK-LABEL: func.func @min_1_input
  // CHECK-NOT: onnx.Min
  // CHECK-NOT: hip.min
  // CHECK: return

  // --- Case 4: dynamic shape ---
  func.func @min_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %result = "onnx.Min"(%a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @min_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<?x?xf32>, %[[B:.*]]: tensor<?x?xf32>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x?xf32>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
  // CHECK: hip.min(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf32>)

  // --- Case 5: rank-5 against a rank-0 scalar, packed to the 4-D path ---
  func.func @min_5d_scalar(%a: tensor<4x1x6x8x1xf32>, %b: tensor<f32>)
      -> tensor<4x1x6x8x1xf32> {
    // CHECK-LABEL: func.func @min_5d_scalar
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<4x1x6x8x1xf32>, %[[B:.*]]: tensor<f32>)
    // CHECK-NOT: onnx.Min
    // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] : tensor<4x1x6x8x1xf32> into tensor<4x1x6x8xf32>
    // CHECK: hip.min(%[[CTX]]) ins({{.*}}, {{.*}} : tensor<4x1x6x8xf32>, tensor<f32>) outs({{.*}} : tensor<4x1x6x8xf32>)
    // CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] output_shape [4, 1, 6, 8, 1] : tensor<4x1x6x8xf32> into tensor<4x1x6x8x1xf32>
    %r = "onnx.Min"(%a, %b) : (tensor<4x1x6x8x1xf32>, tensor<f32>) -> tensor<4x1x6x8x1xf32>
    return %r : tensor<4x1x6x8x1xf32>
  }
}
