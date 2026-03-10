// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ReduceSum operation is correctly converted to HIP reduce_sum operation.
//
// This test validates:
// - onnx.ReduceSum → hip.reduce_sum conversion
// - Axes and keepdims attributes properly handled
// - Context argument inserted via --hip-add-context-arg
// - tensor.empty created for output
//
// Expected: hip.reduce_sum operation with axes and keepdims attributes
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Test 1: Reduce along last axis, keepdims=false
  func.func @reduce_sum_last_axis(%input: tensor<8x128x512xf32>) -> tensor<8x128xf32> {
    // CHECK-LABEL: func.func @reduce_sum_last_axis
    // CHECK-SAME: %[[CTX:.*]]: !hip.context
    // CHECK-SAME: %[[INPUT:.*]]: tensor<8x128x512xf32>
    // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8x128xf32>
    // CHECK: %[[RESULT:.*]] = hip.reduce_sum(%[[CTX]]) ins(%[[INPUT]] : tensor<8x128x512xf32>) outs(%[[EMPTY]] : tensor<8x128xf32>) {axes = [2], keepdims = false} : tensor<8x128xf32>
    // CHECK: return %[[RESULT]]

    %output = "onnx.ReduceSum"(%input) {axes = [2], keepdims = 0 : si64} : (tensor<8x128x512xf32>) -> tensor<8x128xf32>
    return %output : tensor<8x128xf32>
  }

  // Test 2: Reduce along multiple axes, keepdims=true
  func.func @reduce_sum_multi_axes(%input: tensor<8x128x512xf32>) -> tensor<8x1x1xf32> {
    // CHECK-LABEL: func.func @reduce_sum_multi_axes
    // CHECK: hip.reduce_sum(%{{.*}}) ins(%{{.*}} : tensor<8x128x512xf32>) outs(%{{.*}} : tensor<8x1x1xf32>) {axes = [1, 2], keepdims = true} : tensor<8x1x1xf32>

    %output = "onnx.ReduceSum"(%input) {axes = [1, 2], keepdims = 1 : si64} : (tensor<8x128x512xf32>) -> tensor<8x1x1xf32>
    return %output : tensor<8x1x1xf32>
  }

  // Test 3: Reduce along first axis
  func.func @reduce_sum_first_axis(%input: tensor<128x256xf32>) -> tensor<256xf32> {
    // CHECK-LABEL: func.func @reduce_sum_first_axis
    // CHECK: hip.reduce_sum(%{{.*}}) ins(%{{.*}} : tensor<128x256xf32>) outs(%{{.*}} : tensor<256xf32>) {axes = [0], keepdims = false} : tensor<256xf32>

    %output = "onnx.ReduceSum"(%input) {axes = [0], keepdims = 0 : si64} : (tensor<128x256xf32>) -> tensor<256xf32>
    return %output : tensor<256xf32>
  }
}
