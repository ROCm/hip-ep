// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX MatMul is correctly lowered to hip.matmul operation
// in tensor-first mode.
//
// This test validates:
// - MatMul operation lowering (onnx.MatMul → hip.matmul)
// - Batched 3D x 2D matrix multiplication (Llama-3.1 k_proj pattern)
// - Tensor-first DPS: tensor.empty() used as output init
//
// Input: MatMul with 3D input [1x128x4096] and 2D weight [4096x1024]
// Expected: hip.matmul in tensor mode with correct types
// ============================================================================

// RUN: hip-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @matmul_test(%input: tensor<1x128x4096xf16>, %weight: tensor<4096x1024xf16>) -> tensor<1x128x1024xf16> {
    // After conversion: context prepended, tensors remain tensors, tensor return
    // CHECK-LABEL: func.func @matmul_test
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x4096xf16>, %[[WEIGHT:.*]]: tensor<4096x1024xf16>) -> tensor<1x128x1024xf16>

    %output = "onnx.MatMul"(%input, %weight) {onnx_node_name = "/model/layers.0/attn/k_proj/MatMul"} : (tensor<1x128x4096xf16>, tensor<4096x1024xf16>) -> tensor<1x128x1024xf16>

    // After conversion: tensor.empty() for init, hip.matmul in tensor mode
    // CHECK: tensor.empty() : tensor<1x128x1024xf16>
    // CHECK: hip.matmul(%[[CTX]], %[[INPUT]], %[[WEIGHT]], {{.*}})
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<1x128x1024xf16>
  }
}
