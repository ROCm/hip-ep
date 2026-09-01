// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX MatMul is correctly lowered to hip.matmul operation
// in tensor-first mode.
//
// This test validates:
// - MatMul operation lowering (onnx.MatMul → hip.matmul)
// - 2D matrix multiplication (basic case)
// - Batched 3D x 2D matrix multiplication (Llama-3.1 pattern)
// - Dynamic shape support with ?x?x? tensors
// - Tensor-first DPS: tensor.empty() used as output init
// - Proper !hip.context threading through operations
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===== Test 1: From PR-17 (renamed function) =====

  func.func @main_graph(%input: tensor<1x128x4096xf16>, %weight: tensor<4096x1024xf16>) -> tensor<1x128x1024xf16> {
    %output = "onnx.MatMul"(%input, %weight) {onnx_node_name = "/model/layers.0/attn/k_proj/MatMul"} : (tensor<1x128x4096xf16>, tensor<4096x1024xf16>) -> tensor<1x128x1024xf16>
    return %output : tensor<1x128x1024xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x4096xf16>, %[[WEIGHT:.*]]: tensor<4096x1024xf16>) -> tensor<1x128x1024xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x128x1024xf16>
  // CHECK: hip.matmul(%[[CTX]]) ins(%[[INPUT]], %[[WEIGHT]] : tensor<1x128x4096xf16>, tensor<4096x1024xf16>) outs(%[[INIT]] : tensor<1x128x1024xf16>)
  // CHECK-NOT: onnx.MatMul
  // CHECK-NOT: hip.alloc
  // CHECK-NOT: hip.copy

  // ===== Test 2: 3D Batched MatMul =====

  func.func @batched_3d_matmul(%A: tensor<2x128x4096xf16>, %B: tensor<4096x1024xf16>) -> tensor<2x128x1024xf16> {
    %result = "onnx.MatMul"(%A, %B) : (tensor<2x128x4096xf16>, tensor<4096x1024xf16>) -> tensor<2x128x1024xf16>
    return %result : tensor<2x128x1024xf16>
  }

  // CHECK-LABEL: func.func @batched_3d_matmul
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<2x128x4096xf16>, %[[B:.*]]: tensor<4096x1024xf16>) -> tensor<2x128x1024xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x128x1024xf16>
  // CHECK: hip.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<2x128x4096xf16>, tensor<4096x1024xf16>) outs(%[[INIT]] : tensor<2x128x1024xf16>)
  // CHECK-NOT: hip.alloc
  // CHECK-NOT: hip.copy

  // ===== Test 3: Single-axis dynamic batch on both operands =====

  func.func @dynamic_single_axis_batch(
      %A: tensor<?x4x8xf16>, %B: tensor<?x8x16xf16>)
      -> tensor<?x4x16xf16> {
    %result = "onnx.MatMul"(%A, %B)
      : (tensor<?x4x8xf16>, tensor<?x8x16xf16>) -> tensor<?x4x16xf16>
    return %result : tensor<?x4x16xf16>
  }

  // CHECK-LABEL: func.func @dynamic_single_axis_batch
  // CHECK: hip.matmul

  // ===== Test 4: 2D x 3D -- batch comes from B, M comes from A =====
  func.func @matmul_2d_3d(%A: tensor<?x4xf16>, %B: tensor<?x4x?xf16>) -> tensor<?x?x?xf16> {
    %result = "onnx.MatMul"(%A, %B) : (tensor<?x4xf16>, tensor<?x4x?xf16>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }

  // CHECK-LABEL: func.func @matmul_2d_3d
  // CHECK-SAME: (%{{.*}}: !hip.context, %[[A:[A-Za-z0-9_]+]]: tensor<?x4xf16>, %[[B:[A-Za-z0-9_]+]]: tensor<?x4x?xf16>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[BATCH:.*]] = tensor.dim %[[B]], %[[C0]]
  // CHECK-DAG: %[[M:.*]] = tensor.dim %[[A]], %[[C0]]
  // CHECK-DAG: %[[N:.*]] = tensor.dim %[[B]], %[[C2]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[BATCH]], %[[M]], %[[N]]) : tensor<?x?x?xf16>
  // CHECK: hip.matmul

  // ===== Test 5: 2D x 4D -- both batch axes come from B =====
  func.func @matmul_2d_4d(%A: tensor<?x4xf16>, %B: tensor<?x?x4x?xf16>) -> tensor<?x?x?x?xf16> {
    %result = "onnx.MatMul"(%A, %B) : (tensor<?x4xf16>, tensor<?x?x4x?xf16>) -> tensor<?x?x?x?xf16>
    return %result : tensor<?x?x?x?xf16>
  }

  // CHECK-LABEL: func.func @matmul_2d_4d
  // CHECK-SAME: (%{{.*}}: !hip.context, %[[A:[A-Za-z0-9_]+]]: tensor<?x4xf16>, %[[B:[A-Za-z0-9_]+]]: tensor<?x?x4x?xf16>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[B0:.*]] = tensor.dim %[[B]], %[[C0]]
  // CHECK-DAG: %[[B1:.*]] = tensor.dim %[[B]], %[[C1]]
  // CHECK-DAG: %[[M:.*]] = tensor.dim %[[A]], %[[C0]]
  // CHECK-DAG: %[[N:.*]] = tensor.dim %[[B]], %[[C3]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[B0]], %[[B1]], %[[M]], %[[N]]) : tensor<?x?x?x?xf16>
  // CHECK: hip.matmul

  // ===== Test 6: multi-axis dynamic A with rank-2 B =====
  func.func @matmul_4d_2d(
      %A: tensor<?x?x4x8xf16>, %B: tensor<8x16xf16>)
      -> tensor<?x?x4x16xf16> {
    %result = "onnx.MatMul"(%A, %B)
      : (tensor<?x?x4x8xf16>, tensor<8x16xf16>)
        -> tensor<?x?x4x16xf16>
    return %result : tensor<?x?x4x16xf16>
  }

  // CHECK-LABEL: func.func @matmul_4d_2d
  // CHECK: hip.matmul

  // ===== Test 7: explicit single B matrix with multi-axis dynamic A =====
  func.func @matmul_dynamic_broadcast_b(
      %A: tensor<?x?x4x8xf16>, %B: tensor<1x1x8x16xf16>)
      -> tensor<?x?x4x16xf16> {
    %result = "onnx.MatMul"(%A, %B)
      : (tensor<?x?x4x8xf16>, tensor<1x1x8x16xf16>)
        -> tensor<?x?x4x16xf16>
    return %result : tensor<?x?x4x16xf16>
  }

  // CHECK-LABEL: func.func @matmul_dynamic_broadcast_b
  // CHECK: hip.matmul

  // ===== Test 8: explicit single A matrix with multi-axis dynamic B =====
  func.func @matmul_dynamic_broadcast_a(
      %A: tensor<1x1x4x8xf16>, %B: tensor<?x?x8x16xf16>)
      -> tensor<?x?x4x16xf16> {
    %result = "onnx.MatMul"(%A, %B)
      : (tensor<1x1x4x8xf16>, tensor<?x?x8x16xf16>)
        -> tensor<?x?x4x16xf16>
    return %result : tensor<?x?x4x16xf16>
  }

  // CHECK-LABEL: func.func @matmul_dynamic_broadcast_a
  // CHECK: hip.matmul
}
