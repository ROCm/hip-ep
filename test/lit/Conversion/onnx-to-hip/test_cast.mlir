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

  // ==========================================================================
  // ONNX UINT8(2) and BOOL(9) both lower to the MLIR element type `ui8`
  // (see onnxElementTypeToMlirElementType in mlir-imp), so the result type
  // alone cannot disambiguate them. The importer (mlir-graph.cpp) preserves
  // the original ONNX enum in the "onnx_to_enum" attribute; CastToHip must
  // prefer it over re-deriving `to` from the (ambiguous) result type.
  // ==========================================================================

  func.func @cast_f32_to_bool(%input: tensor<4xf32>) -> tensor<4xui8> {
    %output = "onnx.Cast"(%input) {to = ui8, onnx_to_enum = 9 : i64} : (tensor<4xf32>) -> tensor<4xui8>
    return %output : tensor<4xui8>
  }

  // CHECK-LABEL: func.func @cast_f32_to_bool
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xf32>) -> tensor<4xui8>
  // CHECK: tensor.empty() : tensor<4xui8>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xf32>) outs({{.*}} : tensor<4xui8>) {to = 9 : i64}
  // CHECK-NOT: hip.alloc

  func.func @cast_f32_to_uint8(%input: tensor<4xf32>) -> tensor<4xui8> {
    %output = "onnx.Cast"(%input) {to = ui8, onnx_to_enum = 2 : i64} : (tensor<4xf32>) -> tensor<4xui8>
    return %output : tensor<4xui8>
  }

  // CHECK-LABEL: func.func @cast_f32_to_uint8
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xf32>) -> tensor<4xui8>
  // CHECK: tensor.empty() : tensor<4xui8>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xf32>) outs({{.*}} : tensor<4xui8>) {to = 2 : i64}
  // CHECK-NOT: hip.alloc

  // No "onnx_to_enum" attribute (e.g. onnx.Cast synthesized by another pass,
  // or hand-authored IR): falls back to the pre-existing type-based guess,
  // which cannot distinguish UINT8 from BOOL and picks INT8 (3) for any
  // 8-bit integer result. This documents the known fallback limitation
  // rather than silently fixing it, since there is no enum to recover.
  func.func @cast_f32_to_ui8_no_enum_hint(%input: tensor<4xf32>) -> tensor<4xui8> {
    %output = "onnx.Cast"(%input) {to = ui8} : (tensor<4xf32>) -> tensor<4xui8>
    return %output : tensor<4xui8>
  }

  // CHECK-LABEL: func.func @cast_f32_to_ui8_no_enum_hint
  // CHECK: hip.cast({{.*}}) ins({{.*}} : tensor<4xf32>) outs({{.*}} : tensor<4xui8>) {to = 3 : i64}
  // CHECK-NOT: hip.alloc
}
