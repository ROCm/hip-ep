// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX GridSample is lowered to hip.grid_sample.
//
// Test cases:
// 1. Static NCHW bilinear (SimpleBEV-style attrs)
// 2. Nearest + zeros padding
// 3. Dynamic output spatial dims taken from the grid
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // --- Case 1: static bilinear zeros ---
  func.func @grid_sample_bilinear(%X: tensor<1x3x8x8xf32>, %Grid: tensor<1x4x4x2xf32>) -> tensor<1x3x4x4xf32> {
    %Y = "onnx.GridSample"(%X, %Grid) {align_corners = 0 : si64, mode = "bilinear", padding_mode = "zeros"} : (tensor<1x3x8x8xf32>, tensor<1x4x4x2xf32>) -> tensor<1x3x4x4xf32>
    return %Y : tensor<1x3x4x4xf32>
  }

  // CHECK-LABEL: func.func @grid_sample_bilinear
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x3x8x8xf32>, %[[GRID:.*]]: tensor<1x4x4x2xf32>)
  // CHECK-NOT: onnx.GridSample
  // CHECK: tensor.empty() : tensor<1x3x4x4xf32>
  // CHECK: hip.grid_sample(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[GRID]] :
  // CHECK-SAME: {align_corners = 0 : i64, mode = 1 : i64, padding_mode = 0 : i64}

  // --- Case 2: nearest ---
  func.func @grid_sample_nearest(%X: tensor<2x1x4x4xf16>, %Grid: tensor<2x2x2x2xf16>) -> tensor<2x1x2x2xf16> {
    %Y = "onnx.GridSample"(%X, %Grid) {align_corners = 1 : si64, mode = "nearest", padding_mode = "border"} : (tensor<2x1x4x4xf16>, tensor<2x2x2x2xf16>) -> tensor<2x1x2x2xf16>
    return %Y : tensor<2x1x2x2xf16>
  }

  // CHECK-LABEL: func.func @grid_sample_nearest
  // CHECK-NOT: onnx.GridSample
  // CHECK: hip.grid_sample(%{{[^)]*}})
  // CHECK-SAME: {align_corners = 1 : i64, mode = 0 : i64, padding_mode = 1 : i64}

  // --- Case 3: dynamic output spatial from grid ---
  func.func @grid_sample_dynamic(%X: tensor<1x3x8x8xf32>, %Grid: tensor<1x?x?x2xf32>) -> tensor<1x3x?x?xf32> {
    %Y = "onnx.GridSample"(%X, %Grid) {align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<1x3x8x8xf32>, tensor<1x?x?x2xf32>) -> tensor<1x3x?x?xf32>
    return %Y : tensor<1x3x?x?xf32>
  }

  // CHECK-LABEL: func.func @grid_sample_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x3x8x8xf32>, %[[GRID:.*]]: tensor<1x?x?x2xf32>
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[GRID]], %[[C1]]
  // CHECK: %[[DIM2:.*]] = tensor.dim %[[GRID]], %[[C2]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM1]], %[[DIM2]]) : tensor<1x3x?x?xf32>
  // CHECK: hip.grid_sample(%[[CTX]])
  // CHECK-SAME: outs(%[[INIT]] :
}
