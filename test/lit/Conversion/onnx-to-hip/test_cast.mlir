// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Cast is correctly lowered to hip.cast.
//
// Test cases:
// 1. i64 → i32
// 2. f32 → f16
// 3. f16 → f32
// 4. f32 → i64
// 5. i32 → f32
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @cast_i64_to_i32(%input: tensor<4xi64>) -> tensor<4xi32> {
    %output = "onnx.Cast"(%input) {to = i32} : (tensor<4xi64>) -> tensor<4xi32>
    return %output : tensor<4xi32>
  }

  // CHECK-LABEL: func.func @cast_i64_to_i32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xi64>) -> tensor<4xi32>
  // CHECK: tensor.empty() : tensor<4xi32>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xi64>) outs({{.*}} : tensor<4xi32>) : tensor<4xi32>
  // CHECK-NOT: hip.alloc

  func.func @cast_f32_to_f16(%input: tensor<4xf32>) -> tensor<4xf16> {
    %output = "onnx.Cast"(%input) {to = f16} : (tensor<4xf32>) -> tensor<4xf16>
    return %output : tensor<4xf16>
  }

  // CHECK-LABEL: func.func @cast_f32_to_f16
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xf32>) -> tensor<4xf16>
  // CHECK: tensor.empty() : tensor<4xf16>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xf32>) outs({{.*}} : tensor<4xf16>) : tensor<4xf16>
  // CHECK-NOT: hip.alloc

  func.func @cast_f16_to_f32(%input: tensor<4xf16>) -> tensor<4xf32> {
    %output = "onnx.Cast"(%input) {to = f32} : (tensor<4xf16>) -> tensor<4xf32>
    return %output : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @cast_f16_to_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xf16>) -> tensor<4xf32>
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xf16>) outs({{.*}} : tensor<4xf32>) : tensor<4xf32>
  // CHECK-NOT: hip.alloc

  func.func @cast_f32_to_i64(%input: tensor<4xf32>) -> tensor<4xi64> {
    %output = "onnx.Cast"(%input) {to = i64} : (tensor<4xf32>) -> tensor<4xi64>
    return %output : tensor<4xi64>
  }

  // CHECK-LABEL: func.func @cast_f32_to_i64
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xf32>) -> tensor<4xi64>
  // CHECK: tensor.empty() : tensor<4xi64>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xf32>) outs({{.*}} : tensor<4xi64>) : tensor<4xi64>
  // CHECK-NOT: hip.alloc

  func.func @cast_i32_to_f32(%input: tensor<4xi32>) -> tensor<4xf32> {
    %output = "onnx.Cast"(%input) {to = f32} : (tensor<4xi32>) -> tensor<4xf32>
    return %output : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @cast_i32_to_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xi32>) -> tensor<4xf32>
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xi32>) outs({{.*}} : tensor<4xf32>) : tensor<4xf32>
  // CHECK-NOT: hip.alloc
}
