// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the end-to-end CastLike lowering: simplify-onnx rewrites
// onnx.CastLike to plain onnx.Cast and drops the now-dead type-donor
// function argument, hip-add-context-arg then injects !hip.context as arg 0,
// and convert-onnx-to-hip lowers onnx.Cast to hip.cast via the shared
// CastConversion pattern. The target dtype is statically known from the
// CastLike result type, so no new Hip op is needed.
//
// (The simplify-onnx pass is also tested in isolation in
// test_simplify_onnx.mlir; this test covers the full chain to hip.cast.)
//
// Test cases:
// 1. f32 → f16        (basic float narrowing; type-donor arg dropped)
// 2. f16 → f32        (basic float widening; type-donor arg dropped)
// 3. f32 → f32        (identity — should be eliminated, no hip.cast)
// 4. dynamic shape     (f32 → f16 with ?-dims; type-donor arg dropped)
// ============================================================================

// RUN: hip-mlir-opt --simplify-onnx --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // --- Case 1: f32 → f16 ---
  func.func @castlike_f32_to_f16(%input: tensor<3x4xf32>, %target: tensor<0xf16>) -> tensor<3x4xf16> {
    %result = "onnx.CastLike"(%input, %target) : (tensor<3x4xf32>, tensor<0xf16>) -> tensor<3x4xf16>
    return %result : tensor<3x4xf16>
  }

  // The dead %target argument is dropped; the remaining signature is (ctx, input).
  // CHECK-LABEL: func.func @castlike_f32_to_f16
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<3x4xf32>) -> tensor<3x4xf16>
  // CHECK-NOT: onnx.CastLike
  // CHECK: tensor.empty() : tensor<3x4xf16>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<3x4xf32>) outs({{.*}} : tensor<3x4xf16>) {to = 10 : i64} : tensor<3x4xf16>

  // --- Case 2: f16 → f32 ---
  func.func @castlike_f16_to_f32(%input: tensor<4xf16>, %target: tensor<0xf32>) -> tensor<4xf32> {
    %result = "onnx.CastLike"(%input, %target) : (tensor<4xf16>, tensor<0xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @castlike_f16_to_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xf16>) -> tensor<4xf32>
  // CHECK-NOT: onnx.CastLike
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<4xf16>) outs({{.*}} : tensor<4xf32>) {to = 1 : i64} : tensor<4xf32>

  // --- Case 3: identity (f32 → f32) — should be eliminated entirely ---
  func.func @castlike_identity(%input: tensor<4xf32>, %target: tensor<0xf32>) -> tensor<4xf32> {
    %result = "onnx.CastLike"(%input, %target) : (tensor<4xf32>, tensor<0xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @castlike_identity
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NOT: onnx.CastLike
  // CHECK-NOT: hip.cast
  // CHECK: return

  // --- Case 4: dynamic shape f32 → f16 ---
  func.func @castlike_dynamic(%input: tensor<?x?xf32>, %target: tensor<0xf16>) -> tensor<?x?xf16> {
    %result = "onnx.CastLike"(%input, %target) : (tensor<?x?xf32>, tensor<0xf16>) -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }

  // CHECK-LABEL: func.func @castlike_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<?x?xf32>) -> tensor<?x?xf16>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x?xf32>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf16>
  // CHECK: hip.cast(%[[CTX]]) ins(%[[IN]] : tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf16>) {to = 10 : i64} : tensor<?x?xf16>
}
