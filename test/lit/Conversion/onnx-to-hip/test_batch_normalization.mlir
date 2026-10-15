// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX BatchNormalization inference is lowered to hip.batch_norm.
//
// Test cases:
// 1. Static NCHW f16, the shape used by fused image-pooling BN
// 2. Rank-3 (N, C, D) f32
// 3. Dynamic spatial dims
// 4. training_mode stays onnx.BatchNormalization
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // --- Case 1: NCHW f16, spatial 1x1 ---
  func.func @batch_norm_nchw(%X: tensor<1x4x1x1xf16>, %Scale: tensor<4xf16>, %B: tensor<4xf16>, %Mean: tensor<4xf16>, %Var: tensor<4xf16>) -> tensor<1x4x1x1xf16> {
    %Y = "onnx.BatchNormalization"(%X, %Scale, %B, %Mean, %Var) {epsilon = 1.001000e-05 : f32, momentum = 0.899999976 : f32} : (tensor<1x4x1x1xf16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) -> tensor<1x4x1x1xf16>
    return %Y : tensor<1x4x1x1xf16>
  }

  // CHECK-LABEL: func.func @batch_norm_nchw
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x4x1x1xf16>, %[[SCALE:.*]]: tensor<4xf16>, %[[B:.*]]: tensor<4xf16>, %[[MEAN:.*]]: tensor<4xf16>, %[[VAR:.*]]: tensor<4xf16>)
  // CHECK-NOT: onnx.BatchNormalization
  // CHECK: tensor.empty() : tensor<1x4x1x1xf16>
  // CHECK: hip.batch_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[SCALE]], %[[B]], %[[MEAN]], %[[VAR]] :

  // --- Case 2: rank-3 f32 ---
  func.func @batch_norm_rank3(%X: tensor<2x3x8xf32>, %Scale: tensor<3xf32>, %B: tensor<3xf32>, %Mean: tensor<3xf32>, %Var: tensor<3xf32>) -> tensor<2x3x8xf32> {
    %Y = "onnx.BatchNormalization"(%X, %Scale, %B, %Mean, %Var) {epsilon = 1.000000e-05 : f32} : (tensor<2x3x8xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x8xf32>
    return %Y : tensor<2x3x8xf32>
  }

  // CHECK-LABEL: func.func @batch_norm_rank3
  // CHECK-NOT: onnx.BatchNormalization
  // CHECK: hip.batch_norm(%{{[^)]*}})

  // --- Case 3: dynamic spatial ---
  func.func @batch_norm_dynamic(%X: tensor<1x3x?x?xf16>, %Scale: tensor<3xf16>, %B: tensor<3xf16>, %Mean: tensor<3xf16>, %Var: tensor<3xf16>) -> tensor<1x3x?x?xf16> {
    %Y = "onnx.BatchNormalization"(%X, %Scale, %B, %Mean, %Var) {epsilon = 1.000000e-05 : f32} : (tensor<1x3x?x?xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>) -> tensor<1x3x?x?xf16>
    return %Y : tensor<1x3x?x?xf16>
  }

  // CHECK-LABEL: func.func @batch_norm_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x3x?x?xf16>
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[DIM2:.*]] = tensor.dim %[[X]], %[[C2]]
  // CHECK: %[[DIM3:.*]] = tensor.dim %[[X]], %[[C3]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM2]], %[[DIM3]]) : tensor<1x3x?x?xf16>
  // CHECK: hip.batch_norm(%[[CTX]])
  // CHECK-SAME: outs(%[[INIT]] :

  // --- Case 4: training mode is left unconverted ---
  func.func @batch_norm_training(%X: tensor<1x4x2x2xf32>, %Scale: tensor<4xf32>, %B: tensor<4xf32>, %Mean: tensor<4xf32>, %Var: tensor<4xf32>) -> (tensor<1x4x2x2xf32>, tensor<4xf32>, tensor<4xf32>) {
    %Y, %RM, %RV = "onnx.BatchNormalization"(%X, %Scale, %B, %Mean, %Var) {epsilon = 1.000000e-05 : f32, training_mode = 1 : si64} : (tensor<1x4x2x2xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<1x4x2x2xf32>, tensor<4xf32>, tensor<4xf32>)
    return %Y, %RM, %RV : tensor<1x4x2x2xf32>, tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: func.func @batch_norm_training
  // CHECK: onnx.BatchNormalization
  // CHECK-NOT: hip.batch_norm
}
