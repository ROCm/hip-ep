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
// 5. Variadic pairwise broadcast grows the intermediate shape
// 6. Static final result refines dynamic pairwise inference
// 7. Single-input identity preserves a static result refinement
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
  // CHECK-DAG: %[[A0:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x?xf32>
  // CHECK-DAG: %[[A1:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<?x?xf32>
  // CHECK-NOT: arith.select
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[A0]], %[[A1]]) : tensor<?x?xf32>
  // CHECK: hip.min(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf32>)

  // --- Case 5: each pair uses its own broadcast shape ---
  func.func @min_variadic_pairwise_broadcast(%a: tensor<2x1xf32>, %b: tensor<1x3xf32>, %c: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %result = "onnx.Min"(%a, %b, %c) : (tensor<2x1xf32>, tensor<1x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }

  // CHECK-LABEL: func.func @min_variadic_pairwise_broadcast
  // CHECK: %[[E0:.*]] = tensor.empty() : tensor<2x3xf32>
  // CHECK: %[[M0:.*]] = hip.min{{.*}}outs(%[[E0]] : tensor<2x3xf32>)
  // CHECK: %[[E1:.*]] = tensor.empty() : tensor<2x3xf32>
  // CHECK: hip.min{{.*}}ins(%[[M0]], {{.*}} : tensor<2x3xf32>, tensor<2x3xf32>) outs(%[[E1]] : tensor<2x3xf32>)

  // --- Case 6: static final result refines dynamic pairwise inference ---
  func.func @min_variadic_static_refinement(%a: tensor<?x1xf32>, %b: tensor<1x?xf32>, %c: tensor<?x?xf32>) -> tensor<2x3xf32> {
    %result = "onnx.Min"(%a, %b, %c) : (tensor<?x1xf32>, tensor<1x?xf32>, tensor<?x?xf32>) -> tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }

  // CHECK-LABEL: func.func @min_variadic_static_refinement
  // CHECK: %[[E0:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xf32>
  // CHECK: %[[M0:.*]] = hip.min{{.*}}outs(%[[E0]] : tensor<?x?xf32>)
  // CHECK: %[[E1:.*]] = tensor.empty() : tensor<2x3xf32>
  // CHECK: hip.min{{.*}}ins(%[[M0]], {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[E1]] : tensor<2x3xf32>)

  // --- Case 7: single-input identity keeps the imported refinement ---
  func.func @min_single_static_refinement(%a: tensor<?xf32>) -> tensor<7xf32> {
    %result = "onnx.Min"(%a) : (tensor<?xf32>) -> tensor<7xf32>
    return %result : tensor<7xf32>
  }

  // CHECK-LABEL: func.func @min_single_static_refinement
  // CHECK-NOT: onnx.Min
  // CHECK-NOT: hip.min
  // CHECK: tensor.cast %{{.*}} : tensor<?xf32> to tensor<7xf32>
}
