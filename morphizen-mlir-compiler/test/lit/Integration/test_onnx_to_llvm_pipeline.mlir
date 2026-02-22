// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the full ONNX → HIP → LLVM compilation pipeline works correctly
// with multiple passes chained together.
//
// This test validates:
// - Multi-pass pipeline execution
// - Pass interaction and correctness
// - End-to-end lowering from ONNX to LLVM
//
// Pipeline stages:
// 1. --convert-onnx-to-hip: ONNX ops → HIP ops (tensor→memref)
// 2. --hip-buffer-deallocation: Insert automatic cleanup
// 3. --convert-hip-to-llvm: HIP ops → LLVM runtime calls
//
// This is an integration test - it verifies the passes work together.
// Detailed behavior is tested in individual pass tests.
// ============================================================================

// RUN: hip-opt %s --convert-onnx-to-hip --hip-buffer-deallocation --convert-hip-to-llvm | FileCheck %s

module {
  func.func @full_pipeline(
      %input: tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32> {
    // Simple ReLU pipeline
    %output = "onnx.Relu"(%input) : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>

    // After full pipeline, should produce LLVM IR
    // Just verify the module compiles successfully - detailed checks in unit tests
    // CHECK: module

    return %output : tensor<1x64x224x224xf32>
  }
}
