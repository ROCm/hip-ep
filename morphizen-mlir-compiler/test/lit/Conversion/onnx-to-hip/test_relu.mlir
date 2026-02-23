// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ReLU (Rectified Linear Unit) activation is correctly lowered
// to hip.relu operation.
//
// This test validates:
// - Basic activation function lowering (onnx.Relu → hip.relu)
// - Element-wise operation handling
// - In-place operation semantics (input → output)
// - Proper !hip.context threading through operations
//
// Input: ONNX ReLU on 4D tensor (batch x channels x height x width)
// Expected: hip.relu operation preserving tensor shape and context
// ============================================================================

// RUN: morphizen-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @test_relu(%input: tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32> {
    // After conversion, function signature should be transformed:
    // - Context parameter added as first arg
    // - Tensor inputs converted to memrefs with address space 1 (GPU)
    // - Output parameter added (destination-passing style)
    // - Return type changed to i32 (status code)
    // CHECK-LABEL: func.func @test_relu
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: memref<1x64x224x224xf32, 1>, %[[OUTPUT_ARG:.*]]: memref<1x64x224x224xf32, 1>) -> i32

    // ONNX ReLU operation
    %output = "onnx.Relu"(%input) : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>

    // After conversion: should allocate temp buffer, call hip.relu, copy to output
    // CHECK: %[[ALLOC:.*]] = hip.alloc(%[[CTX]]) : memref<1x64x224x224xf32, 1>
    // CHECK-NEXT: hip.relu(%[[CTX]], %[[INPUT]], %[[ALLOC]]) : (!hip.context, memref<1x64x224x224xf32, 1>, memref<1x64x224x224xf32, 1>)
    // CHECK-NEXT: hip.copy(%[[CTX]], %[[ALLOC]], %[[OUTPUT_ARG]]) : (!hip.context, memref<1x64x224x224xf32, 1>, memref<1x64x224x224xf32, 1>)
    // CHECK-NEXT: %[[STATUS:.*]] = arith.constant 0 : i32
    // CHECK-NEXT: return %[[STATUS]] : i32

    return %output : tensor<1x64x224x224xf32>
  }
}
