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

  // ===== Test 3: Dynamic Shapes =====

  func.func @dynamic_matmul(%A: tensor<?x?x?xf16>, %B: tensor<?x?xf16>) -> tensor<?x?x?xf16> {
    %result = "onnx.MatMul"(%A, %B) : (tensor<?x?x?xf16>, tensor<?x?xf16>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }

  // CHECK-LABEL: func.func @dynamic_matmul
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<?x?x?xf16>, %[[B:.*]]: tensor<?x?xf16>) -> tensor<?x?x?xf16>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[D0:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x?x?xf16>
  // CHECK: %[[D1:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<?x?x?xf16>
  // CHECK: %[[D2:.*]] = tensor.dim %[[B]], %[[C1]] : tensor<?x?xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]]) : tensor<?x?x?xf16>
  // CHECK: hip.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%[[INIT]] : tensor<?x?x?xf16>)
  // CHECK-NOT: hip.alloc
  // CHECK-NOT: hip.copy

  // ===== Test 4: A without the dynamic batch dim =====
  // The destination takes its batch dim from the operand that carries it, as
  // hip.matmul's result-shape reification does. It used to be dim 0 of A for
  // every batch dim, which is A's row count for a rank-2 A.

  func.func @shared_rank2_a_dynamic_b_batch(%A: tensor<64x32xf32>, %B: tensor<?x32x49xf32>) -> tensor<?x64x49xf32> {
    %result = "onnx.MatMul"(%A, %B) : (tensor<64x32xf32>, tensor<?x32x49xf32>) -> tensor<?x64x49xf32>
    return %result : tensor<?x64x49xf32>
  }

  // CHECK-LABEL: func.func @shared_rank2_a_dynamic_b_batch
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<64x32xf32>, %[[B:.*]]: tensor<?x32x49xf32>)
  // CHECK-NOT: tensor.dim %[[A]]
  // CHECK: %[[BATCH:.*]] = tensor.dim %[[B]], %{{.*}} : tensor<?x32x49xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[BATCH]]) : tensor<?x64x49xf32>
  // CHECK: hip.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<64x32xf32>, tensor<?x32x49xf32>) outs(%[[INIT]] : tensor<?x64x49xf32>)

  // ===== Test 5: A with a unit batch dim, B with a dynamic batch =====

  func.func @unit_batch_a_dynamic_b_batch(%A: tensor<1x64x32xf32>, %B: tensor<?x32x49xf32>) -> tensor<?x64x49xf32> {
    %result = "onnx.MatMul"(%A, %B) : (tensor<1x64x32xf32>, tensor<?x32x49xf32>) -> tensor<?x64x49xf32>
    return %result : tensor<?x64x49xf32>
  }

  // CHECK-LABEL: func.func @unit_batch_a_dynamic_b_batch
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<1x64x32xf32>, %[[B:.*]]: tensor<?x32x49xf32>)
  // CHECK-NOT: tensor.dim %[[A]]
  // CHECK: %[[BATCH:.*]] = tensor.dim %[[B]], %{{.*}} : tensor<?x32x49xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[BATCH]]) : tensor<?x64x49xf32>
  // CHECK: hip.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<1x64x32xf32>, tensor<?x32x49xf32>) outs(%[[INIT]] : tensor<?x64x49xf32>)

  // ===== Test 6: A carries the dynamic batch, B is shared (unchanged) =====

  func.func @dynamic_a_batch_shared_b(%A: tensor<?x64x32xf32>, %B: tensor<32x49xf32>) -> tensor<?x64x49xf32> {
    %result = "onnx.MatMul"(%A, %B) : (tensor<?x64x32xf32>, tensor<32x49xf32>) -> tensor<?x64x49xf32>
    return %result : tensor<?x64x49xf32>
  }

  // CHECK-LABEL: func.func @dynamic_a_batch_shared_b
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<?x64x32xf32>, %[[B:.*]]: tensor<32x49xf32>)
  // CHECK: %[[BATCH:.*]] = tensor.dim %[[A]], %{{.*}} : tensor<?x64x32xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[BATCH]]) : tensor<?x64x49xf32>
  // CHECK: hip.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x64x32xf32>, tensor<32x49xf32>) outs(%[[INIT]] : tensor<?x64x49xf32>)
}
