// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX InstanceNormalization is lowered to hip.instance_norm.
//
// Test cases:
// 1. Static NCHW f32
// 2. Rank-3 (N, C, D) f16
// 3. Dynamic spatial dims
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // --- Case 1: NCHW f32 ---
  func.func @instance_norm_nchw(%X: tensor<2x3x8x8xf32>, %Scale: tensor<3xf32>, %B: tensor<3xf32>) -> tensor<2x3x8x8xf32> {
    %Y = "onnx.InstanceNormalization"(%X, %Scale, %B) {epsilon = 9.99999974E-6 : f32} : (tensor<2x3x8x8xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x8x8xf32>
    return %Y : tensor<2x3x8x8xf32>
  }

  // CHECK-LABEL: func.func @instance_norm_nchw
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x3x8x8xf32>, %[[SCALE:.*]]: tensor<3xf32>, %[[B:.*]]: tensor<3xf32>)
  // CHECK-NOT: onnx.InstanceNormalization
  // CHECK: tensor.empty() : tensor<2x3x8x8xf32>
  // CHECK: hip.instance_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[SCALE]], %[[B]] :

  // --- Case 2: rank-3 f16 ---
  func.func @instance_norm_rank3(%X: tensor<1x4x16xf16>, %Scale: tensor<4xf16>, %B: tensor<4xf16>) -> tensor<1x4x16xf16> {
    %Y = "onnx.InstanceNormalization"(%X, %Scale, %B) {epsilon = 1.000000e-05 : f32} : (tensor<1x4x16xf16>, tensor<4xf16>, tensor<4xf16>) -> tensor<1x4x16xf16>
    return %Y : tensor<1x4x16xf16>
  }

  // CHECK-LABEL: func.func @instance_norm_rank3
  // CHECK-NOT: onnx.InstanceNormalization
  // CHECK: hip.instance_norm(%{{[^)]*}})

  // --- Case 3: dynamic spatial ---
  func.func @instance_norm_dynamic(%X: tensor<1x3x?x?xf16>, %Scale: tensor<3xf16>, %B: tensor<3xf16>) -> tensor<1x3x?x?xf16> {
    %Y = "onnx.InstanceNormalization"(%X, %Scale, %B) {epsilon = 1.000000e-05 : f32} : (tensor<1x3x?x?xf16>, tensor<3xf16>, tensor<3xf16>) -> tensor<1x3x?x?xf16>
    return %Y : tensor<1x3x?x?xf16>
  }

  // CHECK-LABEL: func.func @instance_norm_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x3x?x?xf16>
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[DIM2:.*]] = tensor.dim %[[X]], %[[C2]]
  // CHECK: %[[DIM3:.*]] = tensor.dim %[[X]], %[[C3]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM2]], %[[DIM3]]) : tensor<1x3x?x?xf16>
  // CHECK: hip.instance_norm(%[[CTX]])
  // CHECK-SAME: outs(%[[INIT]] :
}
