// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ReLU (Rectified Linear Unit) activation is correctly lowered
// to hip.relu operation in tensor-first mode.
//
// This test validates:
// - Basic activation function lowering (onnx.Relu → hip.relu)
// - Element-wise operation handling
// - Tensor-first DPS: tensor.empty() used as output init
// - Proper !hip.context threading through operations
//
// Input: ONNX ReLU on 4D tensor (batch x channels x height x width)
// Expected: hip.relu in tensor mode, returning result tensor directly
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @test_relu(%input: tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32> {
    // After conversion: context prepended, tensors remain tensors, tensor return
    // CHECK-LABEL: func.func @test_relu
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>

    // ONNX ReLU operation
    %output = "onnx.Relu"(%input) : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>

    // After conversion: tensor.empty() for init, hip.relu in tensor mode, return result
    // CHECK: tensor.empty() : tensor<1x64x224x224xf32>
    // CHECK: hip.relu(%[[CTX]], %[[INPUT]], {{.*}})
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<1x64x224x224xf32>
  }
}
