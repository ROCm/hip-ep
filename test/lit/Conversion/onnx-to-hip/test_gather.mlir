// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Gather is correctly lowered to hip.gather operation
// in tensor-first mode.
//
// This test validates:
// - Gather operation lowering (onnx.Gather -> hip.gather)
// - axis = 0: gather along the first axis
// - axis = -1: gather along the last axis (negative indexing)
// - i64 element type support
// - Scalar index and 1D data handling
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
//
// Model: Llama-3.1-8B attention mask shape extraction
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // axis = 0: gather along the first (and only) axis
  func.func @test_gather(%data: tensor<2xi64>, %indices: tensor<i64>) -> tensor<i64> {
    // After conversion: context prepended, tensors remain tensors, tensor return
    // CHECK-LABEL: func.func @test_gather
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<2xi64>, %[[INDICES:.*]]: tensor<i64>) -> tensor<i64>

    %output = "onnx.Gather"(%data, %indices) {axis = 0 : si64} : (tensor<2xi64>, tensor<i64>) -> tensor<i64>

    // After conversion: tensor.empty() for init, hip.gather in tensor mode
    // CHECK: tensor.empty() : tensor<i64>
    // CHECK: hip.gather(%[[CTX]], %[[DATA]], %[[INDICES]], {{.*}}) {axis = 0 : i64}
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<i64>
  }

  // axis = -1: gather along the last axis (negative indexing forwarded as-is)
  func.func @test_gather_neg_axis(%data: tensor<4x8xi64>, %indices: tensor<2xi64>) -> tensor<4x2xi64> {
    // CHECK-LABEL: func.func @test_gather_neg_axis
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<4x8xi64>, %[[INDICES:.*]]: tensor<2xi64>) -> tensor<4x2xi64>

    %output = "onnx.Gather"(%data, %indices) {axis = -1 : si64} : (tensor<4x8xi64>, tensor<2xi64>) -> tensor<4x2xi64>

    // CHECK: tensor.empty() : tensor<4x2xi64>
    // CHECK: hip.gather(%[[CTX]], %[[DATA]], %[[INDICES]], {{.*}}) {axis = -1 : i64}
    // CHECK-NOT: hip.alloc

    return %output : tensor<4x2xi64>
  }
}
