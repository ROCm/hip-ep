// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the OutlineOnnxToHipDNN pass correctly wraps supported ONNX ops
// in hip.hipdnn_graph_outline regions while leaving unsupported or
// dynamic-shape ops untouched for the standard ConvertOnnxToHip path.
//
// This test validates:
// - Static onnx.Conv -> hip.hipdnn_graph_outline with region
// - Region has block args matching operand types
// - Region body contains the original onnx.Conv (cloned)
// - hip.yield terminates the region with correct values
// - Dynamic shapes are NOT outlined (left as onnx.Conv)
// - Unsupported ops are NOT outlined (left as onnx.Sigmoid)
// - Multiple convs each get their own outline op
// ============================================================================

// RUN: hip-mlir-opt %s --outline-onnx-to-hipdnn --split-input-file | FileCheck %s

// -----

// Test 1: Static conv is outlined into hip.hipdnn_graph_outline
func.func @test_static_conv(%x: tensor<1x1x8x8xf32>,
                            %w: tensor<1x1x3x3xf32>) -> tensor<1x1x8x8xf32> {
  %0 = "onnx.Conv"(%x, %w) {auto_pad = "NOTSET", group = 1 : si64,
       kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
       : (tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x8x8xf32>
  return %0 : tensor<1x1x8x8xf32>
}

// CHECK-LABEL: func.func @test_static_conv
// CHECK-SAME:  (%[[X:.*]]: tensor<1x1x8x8xf32>, %[[W:.*]]: tensor<1x1x3x3xf32>)

// CHECK:       %[[OUT:.*]] = hip.hipdnn_graph_outline
// CHECK-SAME:      ins(%[[X]], %[[W]] : tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>)
// CHECK-SAME:      -> tensor<1x1x8x8xf32>

// CHECK:       ^bb0(%[[ARG0:.*]]: tensor<1x1x8x8xf32>, %[[ARG1:.*]]: tensor<1x1x3x3xf32>):
// CHECK:         %[[CONV:.*]] = "onnx.Conv"(%[[ARG0]], %[[ARG1]])
// CHECK:         hip.yield %[[CONV]] : tensor<1x1x8x8xf32>

// CHECK:       return %[[OUT]] : tensor<1x1x8x8xf32>
// CHECK-NOT:   "onnx.Conv"

// -----

// Test 2: Dynamic shapes are NOT outlined (stay as onnx.Conv)
func.func @test_dynamic_conv(%x: tensor<?x1x8x8xf32>,
                             %w: tensor<1x1x3x3xf32>) -> tensor<?x1x8x8xf32> {
  %0 = "onnx.Conv"(%x, %w) {auto_pad = "NOTSET", group = 1 : si64,
       kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
       : (tensor<?x1x8x8xf32>, tensor<1x1x3x3xf32>) -> tensor<?x1x8x8xf32>
  return %0 : tensor<?x1x8x8xf32>
}

// CHECK-LABEL: func.func @test_dynamic_conv
// CHECK:       "onnx.Conv"
// CHECK-NOT:   hip.hipdnn_graph_outline

// -----

// Test 3: Unsupported ops are NOT outlined
func.func @test_unsupported_op(%x: tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32> {
  %0 = "onnx.Sigmoid"(%x) : (tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32>
  return %0 : tensor<1x1x8x8xf32>
}

// CHECK-LABEL: func.func @test_unsupported_op
// CHECK:       "onnx.Sigmoid"
// CHECK-NOT:   hip.hipdnn_graph_outline

// -----

// Test 4: Multiple static convs each get their own outline op
func.func @test_multiple_convs(%x: tensor<1x1x8x8xf32>,
                               %w1: tensor<1x1x3x3xf32>,
                               %w2: tensor<1x1x3x3xf32>) -> tensor<1x1x8x8xf32> {
  %0 = "onnx.Conv"(%x, %w1) {auto_pad = "NOTSET", group = 1 : si64,
       kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
       : (tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x8x8xf32>
  %1 = "onnx.Conv"(%0, %w2) {auto_pad = "NOTSET", group = 1 : si64,
       kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
       : (tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x8x8xf32>
  return %1 : tensor<1x1x8x8xf32>
}

// CHECK-LABEL: func.func @test_multiple_convs
// CHECK:       hip.hipdnn_graph_outline
// CHECK:         "onnx.Conv"
// CHECK:         hip.yield
// CHECK:       hip.hipdnn_graph_outline
// CHECK:         "onnx.Conv"
// CHECK:         hip.yield
// CHECK-NOT:   "onnx.Conv"

// -----

// Test 5: Hybrid model -- Conv is outlined, Sigmoid stays as onnx op
func.func @test_hybrid(%x: tensor<1x1x8x8xf32>,
                       %w: tensor<1x1x3x3xf32>) -> tensor<1x1x8x8xf32> {
  %0 = "onnx.Conv"(%x, %w) {auto_pad = "NOTSET", group = 1 : si64,
       kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
       : (tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x8x8xf32>
  %1 = "onnx.Sigmoid"(%0) : (tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32>
  return %1 : tensor<1x1x8x8xf32>
}

// CHECK-LABEL: func.func @test_hybrid
// CHECK:       %[[CONV:.*]] = hip.hipdnn_graph_outline
// CHECK:         "onnx.Conv"
// CHECK:         hip.yield
// CHECK:       "onnx.Sigmoid"(%[[CONV]])
// CHECK:       return
