// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Cast operation is correctly converted to HIP cast operation.
//
// This test validates:
// - onnx.Cast → hip.cast conversion
// - Type attribute properly handled
// - Context argument inserted via --hip-add-context-arg
// - tensor.empty created for output
//
// Expected: hip.cast operation with same input/output shapes but different types
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Test 1: f32 -> f16 conversion (2D)
  func.func @cast_f32_to_f16(%input: tensor<128x256xf32>) -> tensor<128x256xf16> {
    // CHECK-LABEL: func.func @cast_f32_to_f16
    // CHECK-SAME: %[[CTX:.*]]: !hip.context
    // CHECK-SAME: %[[INPUT:.*]]: tensor<128x256xf32>
    // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<128x256xf16>
    // CHECK: %[[RESULT:.*]] = hip.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<128x256xf32>) outs(%[[EMPTY]] : tensor<128x256xf16>) : tensor<128x256xf16>
    // CHECK: return %[[RESULT]]

    %output = "onnx.Cast"(%input) {to = f16} : (tensor<128x256xf32>) -> tensor<128x256xf16>
    return %output : tensor<128x256xf16>
  }

  // Test 2: f16 -> f32 conversion (3D)
  func.func @cast_f16_to_f32(%input: tensor<8x128x512xf16>) -> tensor<8x128x512xf32> {
    // CHECK-LABEL: func.func @cast_f16_to_f32
    // CHECK: hip.cast(%{{.*}}) ins(%{{.*}} : tensor<8x128x512xf16>) outs(%{{.*}} : tensor<8x128x512xf32>) : tensor<8x128x512xf32>

    %output = "onnx.Cast"(%input) {to = f32} : (tensor<8x128x512xf16>) -> tensor<8x128x512xf32>
    return %output : tensor<8x128x512xf32>
  }

  // Test 3: f32 -> i32 conversion (4D)
  func.func @cast_f32_to_i32(%input: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xi32> {
    // CHECK-LABEL: func.func @cast_f32_to_i32
    // CHECK: hip.cast(%{{.*}}) ins(%{{.*}} : tensor<1x64x56x56xf32>) outs(%{{.*}} : tensor<1x64x56x56xi32>) : tensor<1x64x56x56xi32>

    %output = "onnx.Cast"(%input) {to = i32} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xi32>
    return %output : tensor<1x64x56x56xi32>
  }
}
