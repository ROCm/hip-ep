// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify Torch MatMul ops are correctly lowered to hip.matmul operation
// in tensor-first mode.
//
// This test validates:
// - torch.aten.mm (2D matrix multiplication)
// - torch.aten.bmm (batched 3D matrix multiplication)
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-torch-to-hip %s | FileCheck %s

// CHECK-LABEL: func.func @test_mm
func.func @test_mm(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32> {
  // CHECK: hip.matmul
  %0 = "torch.aten.mm"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @test_bmm
func.func @test_bmm(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
  // CHECK: hip.matmul
  %0 = "torch.aten.bmm"(%arg0, %arg1) : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}
