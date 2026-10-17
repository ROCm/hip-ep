// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify a binary contraction einsum lowers to hip.matmul.
//
// 1. bhwc,hkc->bhwk  (batch h, contract c)
// 2. bhwc,wkc->bhwk  (batch w, contract c)
// 3. ij,jk->ik       (plain matmul, no transpose)
// 4. A single-operand reduction stays onnx.Einsum
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @einsum_bhwc_hkc(%A: tensor<2x3x4x5xf16>, %B: tensor<3x6x5xf16>) -> tensor<2x3x4x6xf16> {
    %Y = "onnx.Einsum"(%A, %B) {equation = "bhwc,hkc->bhwk"} : (tensor<2x3x4x5xf16>, tensor<3x6x5xf16>) -> tensor<2x3x4x6xf16>
    return %Y : tensor<2x3x4x6xf16>
  }

  // CHECK-LABEL: func.func @einsum_bhwc_hkc
  // CHECK-NOT: onnx.Einsum
  // CHECK: hip.transpose
  // CHECK: tensor.collapse_shape
  // CHECK: hip.matmul
  // CHECK-SAME: transB = 1
  // CHECK: tensor.expand_shape
  // CHECK: hip.transpose

  func.func @einsum_bhwc_wkc(%A: tensor<2x3x4x5xf16>, %B: tensor<4x6x5xf16>) -> tensor<2x3x4x6xf16> {
    %Y = "onnx.Einsum"(%A, %B) {equation = "bhwc,wkc->bhwk"} : (tensor<2x3x4x5xf16>, tensor<4x6x5xf16>) -> tensor<2x3x4x6xf16>
    return %Y : tensor<2x3x4x6xf16>
  }

  // CHECK-LABEL: func.func @einsum_bhwc_wkc
  // CHECK-NOT: onnx.Einsum
  // CHECK: hip.matmul
  // CHECK-SAME: transB = 1

  func.func @einsum_matmul(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %Y = "onnx.Einsum"(%A, %B) {equation = "ij,jk->ik"} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %Y : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @einsum_matmul
  // CHECK-NOT: onnx.Einsum
  // CHECK-NOT: hip.transpose
  // CHECK: hip.matmul
  // CHECK-NOT: tensor.collapse_shape
  // CHECK-NOT: tensor.expand_shape

  func.func @einsum_reduce(%A: tensor<2x3xf32>) -> tensor<2xf32> {
    %Y = "onnx.Einsum"(%A) {equation = "ij->i"} : (tensor<2x3xf32>) -> tensor<2xf32>
    return %Y : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @einsum_reduce
  // CHECK: onnx.Einsum
  // CHECK-NOT: hip.matmul
}
