// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Add (elementwise addition) is correctly lowered
// to hip.add operation in tensor-first mode.
//
// This test validates:
// - Elementwise binary operation lowering (onnx.Add → hip.add)
// - Same-rank operands (3D + 3D)
// - Broadcasting operands (3D + 1D bias)
// - f16 element type support
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
//
// Model: GPT-OSS-20B MoE router bias add
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Test 1: Same-rank addition (3D + 3D)
  func.func @test_add_same_rank(%lhs: tensor<1x128x32xf16>, %rhs: tensor<1x128x32xf16>) -> tensor<1x128x32xf16> {
    // CHECK-LABEL: func.func @test_add_same_rank
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<1x128x32xf16>, %[[RHS:.*]]: tensor<1x128x32xf16>) -> tensor<1x128x32xf16>

    %output = "onnx.Add"(%lhs, %rhs) : (tensor<1x128x32xf16>, tensor<1x128x32xf16>) -> tensor<1x128x32xf16>

    // CHECK: tensor.empty() : tensor<1x128x32xf16>
    // CHECK: hip.add(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<1x128x32xf16>, tensor<1x128x32xf16>) outs({{.*}} : tensor<1x128x32xf16>)
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<1x128x32xf16>
  }

  // Test 2: Broadcasting addition (3D + 1D bias)
  func.func @test_add_broadcast(%input: tensor<1x128x32xf16>, %bias: tensor<32xf16>) -> tensor<1x128x32xf16> {
    // CHECK-LABEL: func.func @test_add_broadcast
    // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x32xf16>, %[[BIAS:.*]]: tensor<32xf16>) -> tensor<1x128x32xf16>

    %output = "onnx.Add"(%input, %bias) : (tensor<1x128x32xf16>, tensor<32xf16>) -> tensor<1x128x32xf16>

    // CHECK: tensor.empty() : tensor<1x128x32xf16>
    // CHECK: hip.add(%[[CTX2]]) ins(%[[INPUT]], %[[BIAS]] : tensor<1x128x32xf16>, tensor<32xf16>) outs({{.*}} : tensor<1x128x32xf16>)

    return %output : tensor<1x128x32xf16>
  }

  // Test 3: Outer-broadcast dynamic add ([?x1] + [1x?] -> [?x?]).
  // Dim 0 must come from lhs; dim 1 from rhs (not lhs's static 1).
  func.func @test_add_outer_broadcast_dynamic(%lhs: tensor<?x1xi64>,
                                              %rhs: tensor<1x?xi64>)
      -> tensor<?x?xi64> {
    // CHECK-LABEL: func.func @test_add_outer_broadcast_dynamic
    // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[LHS:.*]]: tensor<?x1xi64>, %[[RHS:.*]]: tensor<1x?xi64>) -> tensor<?x?xi64>

    %output = "onnx.Add"(%lhs, %rhs) :
        (tensor<?x1xi64>, tensor<1x?xi64>) -> tensor<?x?xi64>

    // CHECK: %[[D0:.*]] = tensor.dim %[[LHS]], %c0
    // CHECK: %[[D1:.*]] = tensor.dim %[[RHS]], %c1
    // CHECK-NOT: tensor.dim %[[LHS]], %c1
    // CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xi64>
    // CHECK: hip.add(%[[CTX3]]) ins(%[[LHS]], %[[RHS]] : tensor<?x1xi64>, tensor<1x?xi64>) outs(%[[INIT]] : tensor<?x?xi64>) : tensor<?x?xi64>

    return %output : tensor<?x?xi64>
  }

  func.func @main_graph(%arg0: tensor<1x128x32xf16>, %arg1: tensor<1x128x32xf16>) -> tensor<1x128x32xf16> {
    return %arg0 : tensor<1x128x32xf16>
  }
}
