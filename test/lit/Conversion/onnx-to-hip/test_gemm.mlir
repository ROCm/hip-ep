// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Gemm (General Matrix-Matrix Multiplication) is correctly lowered
// to hip.gemm operation in tensor-first mode.
//
// This test validates:
// - Gemm operation lowering (onnx.Gemm → hip.gemm)
// - Three-input operation (A, B, C), where C is optional
// - f16 element type support
// - GEMM-specific attributes: alpha, beta, transA, transB
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
//
// Model: qwen-2.5 FC layer
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {

  // --- Case 1: Gemm with C (bias), default attributes ---
  func.func @test_gemm_with_c(%a: tensor<1x5120xf16>, %b: tensor<5120x5120xf16>, %c: tensor<5120xf16>) -> tensor<1x5120xf16> {
    // CHECK-LABEL: func.func @test_gemm_with_c
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<1x5120xf16>, %[[B:.*]]: tensor<5120x5120xf16>, %[[C:.*]]: tensor<5120xf16>) -> tensor<1x5120xf16>

    %output = "onnx.Gemm"(%a, %b, %c) : (tensor<1x5120xf16>, tensor<5120x5120xf16>, tensor<5120xf16>) -> tensor<1x5120xf16>

    // CHECK: tensor.empty() : tensor<1x5120xf16>
    // CHECK: hip.gemm(%[[CTX]]) ins(%[[A]], %[[B]], %[[C]] : tensor<1x5120xf16>, tensor<5120x5120xf16>, tensor<5120xf16>) outs({{.*}} : tensor<1x5120xf16>)
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<1x5120xf16>
  }

  // --- Case 2: Gemm without C (no bias) ---
  func.func @test_gemm_no_c(%a: tensor<1x5120xf16>, %b: tensor<5120x5120xf16>) -> tensor<1x5120xf16> {
    // CHECK-LABEL: func.func @test_gemm_no_c
    // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[A2:.*]]: tensor<1x5120xf16>, %[[B2:.*]]: tensor<5120x5120xf16>) -> tensor<1x5120xf16>

    %none = "onnx.NoValue"() {value} : () -> none
    %output = "onnx.Gemm"(%a, %b, %none) : (tensor<1x5120xf16>, tensor<5120x5120xf16>, none) -> tensor<1x5120xf16>

    // CHECK: tensor.empty() : tensor<1x5120xf16>
    // CHECK: hip.gemm(%[[CTX2]]) ins(%[[A2]], %[[B2]] : tensor<1x5120xf16>, tensor<5120x5120xf16>) outs({{.*}} : tensor<1x5120xf16>)
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<1x5120xf16>
  }

  // --- Case 3: Gemm with transB = 1 ---
  func.func @test_gemm_transB(%a: tensor<1x5120xf16>, %b: tensor<5120x5120xf16>, %c: tensor<5120xf16>) -> tensor<1x5120xf16> {
    // CHECK-LABEL: func.func @test_gemm_transB
    // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[A3:.*]]: tensor<1x5120xf16>, %[[B3:.*]]: tensor<5120x5120xf16>, %[[C3:.*]]: tensor<5120xf16>) -> tensor<1x5120xf16>

    %output = "onnx.Gemm"(%a, %b, %c) {transB = 1 : si64} : (tensor<1x5120xf16>, tensor<5120x5120xf16>, tensor<5120xf16>) -> tensor<1x5120xf16>

    // CHECK: tensor.empty() : tensor<1x5120xf16>
    // CHECK: hip.gemm(%[[CTX3]]) ins(%[[A3]], %[[B3]], %[[C3]] : tensor<1x5120xf16>, tensor<5120x5120xf16>, tensor<5120xf16>) outs({{.*}} : tensor<1x5120xf16>)
    // CHECK-SAME: transB = 1 : i64
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<1x5120xf16>
  }

  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<1x5120xf16>, %arg1: tensor<5120x5120xf16>, %arg2: tensor<5120xf16>) -> tensor<1x5120xf16> {
    return %arg0 : tensor<1x5120xf16>
  }
}
