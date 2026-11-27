// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics %s | FileCheck %s

// ============================================================================
// TEST PURPOSE:
// The MatMul destination is built from hip::reifyMatmulShape. When the operand
// shapes do not fit a matmul, or contradict the result type the importer
// declared, the conversion declines: the onnx.MatMul stays as it was, and the
// rewrite leaves no hip.matmul, tensor.empty or tensor.dim behind.
//
// The shape rule reports an incompatible pair of operands. A result type that
// only contradicts the inferred shape is declined without a diagnostic, which
// --verify-diagnostics checks by expecting none for it.
// ============================================================================

module {
  // The contraction dims differ: A is [4, 8] and B is [5, 7].
  func.func @main_graph(%a: tensor<4x8xf32>, %b: tensor<5x7xf32>) -> tensor<4x7xf32> {
    // expected-error @+1 {{matmul contraction dim mismatch}}
    %y = "onnx.MatMul"(%a, %b) : (tensor<4x8xf32>, tensor<5x7xf32>) -> tensor<4x7xf32>
    return %y : tensor<4x7xf32>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK: onnx.MatMul
  // CHECK-NOT: hip.matmul
  // CHECK-NOT: tensor.empty
  // CHECK-NOT: tensor.dim

  // The batch dims of A and B do not broadcast: 2 against 3.
  func.func @batch_conflict(%a: tensor<2x4x8xf32>, %b: tensor<3x8x5xf32>) -> tensor<?x4x5xf32> {
    // expected-error @+1 {{matmul batch broadcast failure}}
    %y = "onnx.MatMul"(%a, %b) : (tensor<2x4x8xf32>, tensor<3x8x5xf32>) -> tensor<?x4x5xf32>
    return %y : tensor<?x4x5xf32>
  }

  // CHECK-LABEL: func.func @batch_conflict
  // CHECK: onnx.MatMul
  // CHECK-NOT: hip.matmul
  // CHECK-NOT: tensor.empty
  // CHECK-NOT: tensor.dim

  // The operands fit, but the declared result has N = 6 where B gives 5. The
  // batch dim is dynamic, so a tensor.dim is built before the conflict shows;
  // it must not stay behind.
  func.func @result_contradicts_operands(%a: tensor<?x4x8xf32>, %b: tensor<8x5xf32>) -> tensor<?x4x6xf32> {
    %y = "onnx.MatMul"(%a, %b) : (tensor<?x4x8xf32>, tensor<8x5xf32>) -> tensor<?x4x6xf32>
    return %y : tensor<?x4x6xf32>
  }

  // CHECK-LABEL: func.func @result_contradicts_operands
  // CHECK: onnx.MatMul
  // CHECK-NOT: hip.matmul
  // CHECK-NOT: tensor.empty
  // CHECK-NOT: tensor.dim
}
