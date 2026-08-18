// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the pre-lowering Transpose(last-two-swap) -> MatMul fold and the
// resulting hip.matmul transB attribute for attention Q @ K^T patterns.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1xf16>) -> tensor<1x1xf16> {
    return %arg0 : tensor<1x1xf16>
  }

  // Swin-style 4D batched attention: Q @ K^T with perm=[0,1,3,2].
  func.func @attention_qk(%q: tensor<2x3x8x16xf16>, %k: tensor<2x3x8x16xf16>)
      -> tensor<2x3x8x8xf16> {
    %kt = "onnx.Transpose"(%k) {perm = dense<[0, 1, 3, 2]> : tensor<4xi64>}
        : (tensor<2x3x8x16xf16>) -> tensor<2x3x16x8xf16>
    %scores = "onnx.MatMul"(%q, %kt)
        : (tensor<2x3x8x16xf16>, tensor<2x3x16x8xf16>) -> tensor<2x3x8x8xf16>
    return %scores : tensor<2x3x8x8xf16>
  }

  // CHECK-LABEL: func.func @attention_qk
  // CHECK-NOT: onnx.Transpose
  // CHECK-NOT: onnx.MatMul
  // CHECK: hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<2x3x8x16xf16>, tensor<2x3x8x16xf16>)
  // CHECK-SAME: transB = 1
  // CHECK-NOT: hip.transpose

  // Same fold when perm is an ArrayAttr (MorphiZen / ONNX importer form).
  func.func @attention_qk_array_perm(%q: tensor<2x3x8x16xf16>,
                                     %k: tensor<2x3x8x16xf16>)
      -> tensor<2x3x8x8xf16> {
    %kt = "onnx.Transpose"(%k) {perm = [0, 1, 3, 2]}
        : (tensor<2x3x8x16xf16>) -> tensor<2x3x16x8xf16>
    %scores = "onnx.MatMul"(%q, %kt)
        : (tensor<2x3x8x16xf16>, tensor<2x3x16x8xf16>) -> tensor<2x3x8x8xf16>
    return %scores : tensor<2x3x8x8xf16>
  }

  // CHECK-LABEL: func.func @attention_qk_array_perm
  // CHECK-NOT: onnx.Transpose
  // CHECK: hip.matmul
  // CHECK-SAME: transB = 1

  // transA on the left operand.
  func.func @matmul_trans_a(%a: tensor<4x8xf16>, %b: tensor<4x16xf16>)
      -> tensor<8x16xf16> {
    %at = "onnx.Transpose"(%a) {perm = dense<[1, 0]> : tensor<2xi64>}
        : (tensor<4x8xf16>) -> tensor<8x4xf16>
    %y = "onnx.MatMul"(%at, %b)
        : (tensor<8x4xf16>, tensor<4x16xf16>) -> tensor<8x16xf16>
    return %y : tensor<8x16xf16>
  }

  // CHECK-LABEL: func.func @matmul_trans_a
  // CHECK-NOT: onnx.Transpose
  // CHECK: hip.matmul
  // CHECK-SAME: transA = 1

  // Non-matching perm (full reverse on rank 3) must survive.
  func.func @no_fold_full_reverse(%a: tensor<2x3x4xf16>, %b: tensor<2x3x4xf16>)
      -> tensor<2x3x3xf16> {
    %bt = "onnx.Transpose"(%b) {perm = dense<[2, 1, 0]> : tensor<3xi64>}
        : (tensor<2x3x4xf16>) -> tensor<2x4x3xf16>
    %y = "onnx.MatMul"(%a, %bt)
        : (tensor<2x3x4xf16>, tensor<2x4x3xf16>) -> tensor<2x3x3xf16>
    return %y : tensor<2x3x3xf16>
  }

  // CHECK-LABEL: func.func @no_fold_full_reverse
  // CHECK: hip.transpose
  // CHECK: hip.matmul
  // CHECK-NOT: transB = 1
  // CHECK-NOT: transA = 1
}
