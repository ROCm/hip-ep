// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Cast is correctly lowered to hip.cast with the right ONNX
// DataType enum encoded in the `to` attribute.
//
// ONNX DataType enum values used here:
//   f32 = 1,  i32 = 6,  i64 = 7,  f16 = 10
//
// Test cases:
// 1. i64 → i32   (to = 6)
// 2. f32 → f16   (to = 10)
// 3. f16 → f32   (to = 1)
// 4. f32 → i64   (to = 7)
// 5. i32 → f32   (to = 1)
// 6. i64 → BOOL  (to = i1 on a ui8 result, to = 9)
// 7. i64 → ui8   (to = 3, by value)
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @cast_i64_to_i32(%input: tensor<4xi64>) -> tensor<4xi32> {
    %output = "onnx.Cast"(%input) {to = i32} : (tensor<4xi64>) -> tensor<4xi32>
    return %output : tensor<4xi32>
  }

  // CHECK-LABEL: func.func @cast_i64_to_i32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xi64>) -> tensor<4xi32>
  // CHECK: tensor.empty() : tensor<4xi32>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xi64>) outs({{.*}} : tensor<4xi32>) {to = 6 : i64} : tensor<4xi32>
  // CHECK-NOT: hip.alloc

  func.func @cast_f32_to_f16(%input: tensor<4xf32>) -> tensor<4xf16> {
    %output = "onnx.Cast"(%input) {to = f16} : (tensor<4xf32>) -> tensor<4xf16>
    return %output : tensor<4xf16>
  }

  // CHECK-LABEL: func.func @cast_f32_to_f16
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xf32>) -> tensor<4xf16>
  // CHECK: tensor.empty() : tensor<4xf16>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xf32>) outs({{.*}} : tensor<4xf16>) {to = 10 : i64}
  // CHECK-NOT: hip.alloc

  func.func @cast_f16_to_f32(%input: tensor<4xf16>) -> tensor<4xf32> {
    %output = "onnx.Cast"(%input) {to = f32} : (tensor<4xf16>) -> tensor<4xf32>
    return %output : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @cast_f16_to_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xf16>) -> tensor<4xf32>
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xf16>) outs({{.*}} : tensor<4xf32>) {to = 1 : i64}
  // CHECK-NOT: hip.alloc

  func.func @cast_f32_to_i64(%input: tensor<4xf32>) -> tensor<4xi64> {
    %output = "onnx.Cast"(%input) {to = i64} : (tensor<4xf32>) -> tensor<4xi64>
    return %output : tensor<4xi64>
  }

  // CHECK-LABEL: func.func @cast_f32_to_i64
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xf32>) -> tensor<4xi64>
  // CHECK: tensor.empty() : tensor<4xi64>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xf32>) outs({{.*}} : tensor<4xi64>) {to = 7 : i64}
  // CHECK-NOT: hip.alloc

  func.func @cast_i32_to_f32(%input: tensor<4xi32>) -> tensor<4xf32> {
    %output = "onnx.Cast"(%input) {to = f32} : (tensor<4xi32>) -> tensor<4xf32>
    return %output : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @cast_i32_to_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xi32>) -> tensor<4xf32>
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xi32>) outs({{.*}} : tensor<4xf32>) {to = 1 : i64}
  // CHECK-NOT: hip.alloc

  // Dynamic shape test
  func.func @cast_dynamic_f32_to_f16(%input: tensor<?x?x512xf32>) -> tensor<?x?x512xf16> {
    %output = "onnx.Cast"(%input) {to = f16} : (tensor<?x?x512xf32>) -> tensor<?x?x512xf16>
    return %output : tensor<?x?x512xf16>
  }

  // CHECK-LABEL: func.func @cast_dynamic_f32_to_f16
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<?x?x512xf32>) -> tensor<?x?x512xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x512xf16>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<?x?x512xf32>) outs(%[[INIT]] : tensor<?x?x512xf16>) {to = 10 : i64} : tensor<?x?x512xf16>
  // CHECK-NOT: hip.alloc

  // The importer stores BOOL as ui8 and marks the cast with `to = i1`, which
  // must become `to = 9` (BOOL), not a by-value cast to UINT8.
  func.func @cast_i64_to_bool(%input: tensor<4xi64>) -> tensor<4xui8> {
    %output = "onnx.Cast"(%input) {to = i1} : (tensor<4xi64>) -> tensor<4xui8>
    return %output : tensor<4xui8>
  }

  // CHECK-LABEL: func.func @cast_i64_to_bool
  // CHECK: hip.cast({{.*}}) ins(%{{.*}} : tensor<4xi64>) outs({{.*}} : tensor<4xui8>) {to = 9 : i64}

  // A plain UINT8 target keeps its by-value cast.
  func.func @cast_i64_to_ui8(%input: tensor<4xi64>) -> tensor<4xui8> {
    %output = "onnx.Cast"(%input) {to = ui8} : (tensor<4xi64>) -> tensor<4xui8>
    return %output : tensor<4xui8>
  }

  // CHECK-LABEL: func.func @cast_i64_to_ui8
  // CHECK: hip.cast({{.*}}) ins(%{{.*}} : tensor<4xi64>) outs({{.*}} : tensor<4xui8>) {to = 3 : i64}
}
