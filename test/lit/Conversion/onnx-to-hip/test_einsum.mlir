// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify a binary contraction einsum lowers to hip.matmul.
//
// 1. bhwc,hkc->bhwk  (batch h, contract c)
// 2. bhwc,wkc->bhwk  (batch w, contract c)
// 3. ij,jk->ik       (plain matmul, no transpose)
// 4. bc,d->bcd       (outer product: no contracted index)
// 5. A reordered outer product inserts the unit axes in output order
// 6. Rank 5 stays onnx.Einsum because hip.mul's ABI stops at rank 4
// 7. A single-operand reduction stays onnx.Einsum
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

  func.func @einsum_outer(%A: tensor<1x3xf16>, %B: tensor<10xf16>) -> tensor<1x3x10xf16> {
    %Y = "onnx.Einsum"(%A, %B) {equation = "bc,d->bcd"} : (tensor<1x3xf16>, tensor<10xf16>) -> tensor<1x3x10xf16>
    return %Y : tensor<1x3x10xf16>
  }

  // CHECK-LABEL: func.func @einsum_outer
  // CHECK: tensor.expand_shape
  // CHECK: tensor<1x3x1xf16>
  // CHECK: tensor.expand_shape
  // CHECK: tensor<1x1x10xf16>
  // CHECK: hip.mul
  // CHECK-NOT: onnx.Einsum
  // CHECK-NOT: hip.matmul

  func.func @einsum_outer_reordered(%A: tensor<10xf16>, %B: tensor<1x3xf16>) -> tensor<3x10x1xf16> {
    %Y = "onnx.Einsum"(%A, %B) {equation = "d,bc->cdb"} : (tensor<10xf16>, tensor<1x3xf16>) -> tensor<3x10x1xf16>
    return %Y : tensor<3x10x1xf16>
  }

  // CHECK-LABEL: func.func @einsum_outer_reordered
  // CHECK: tensor.expand_shape
  // CHECK: tensor<1x10x1xf16>
  // CHECK: hip.transpose
  // CHECK: tensor.expand_shape
  // CHECK: tensor<3x1x1xf16>
  // CHECK: hip.mul
  // CHECK-NOT: onnx.Einsum
  // CHECK-NOT: hip.matmul

  func.func @einsum_outer_f64(%A: tensor<1x3xf64>, %B: tensor<10xf64>) -> tensor<1x3x10xf64> {
    %Y = "onnx.Einsum"(%A, %B) {equation = "bc,d->bcd"} : (tensor<1x3xf64>, tensor<10xf64>) -> tensor<1x3x10xf64>
    return %Y : tensor<1x3x10xf64>
  }

  // CHECK-LABEL: func.func @einsum_outer_f64
  // CHECK: onnx.Einsum
  // CHECK-NOT: hip.mul

  func.func @einsum_outer_rank5(%A: tensor<2x2x2x2x2xf16>, %B: tensor<2xf16>) -> tensor<2x2x2x2x2x2xf16> {
    %Y = "onnx.Einsum"(%A, %B) {equation = "abcde,f->abcdef"} : (tensor<2x2x2x2x2xf16>, tensor<2xf16>) -> tensor<2x2x2x2x2x2xf16>
    return %Y : tensor<2x2x2x2x2x2xf16>
  }

  // CHECK-LABEL: func.func @einsum_outer_rank5
  // CHECK: onnx.Einsum
  // CHECK-NOT: hip.mul

  func.func @einsum_reduce(%A: tensor<2x3xf32>) -> tensor<2xf32> {
    %Y = "onnx.Einsum"(%A) {equation = "ij->i"} : (tensor<2x3xf32>) -> tensor<2xf32>
    return %Y : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @einsum_reduce
  // CHECK: onnx.Einsum
  // CHECK-NOT: hip.matmul
}
