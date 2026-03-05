// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Sub (elementwise subtraction) is correctly lowered
// to hip.sub operation in tensor-first mode.
//
// This test validates:
// - Elementwise binary operation lowering (onnx.Sub -> hip.sub)
// - Two-input operand handling
// - i64 element type support
// - 2D tensor shape preservation
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
//
// Model: Llama-3.1-8B attention mask reformatting (ReduceSum - 1)
// ============================================================================

// RUN: udna-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @test_sub(%lhs: tensor<1x1xi64>, %rhs: tensor<1x1xi64>) -> tensor<1x1xi64> {
    // After conversion: context prepended, tensors remain tensors, tensor return
    // CHECK-LABEL: func.func @test_sub
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<1x1xi64>, %[[RHS:.*]]: tensor<1x1xi64>) -> tensor<1x1xi64>

    %output = "onnx.Sub"(%lhs, %rhs) : (tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x1xi64>

    // After conversion: tensor.empty() for init, hip.sub in tensor mode
    // CHECK: tensor.empty() : tensor<1x1xi64>
    // CHECK: hip.sub(%[[CTX]], %[[LHS]], %[[RHS]], {{.*}})
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<1x1xi64>
  }
}
