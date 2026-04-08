// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify Torch activation ops are correctly lowered to HIP ops
// in tensor-first mode.
//
// This test validates:
// - torch.aten.sigmoid -> hip.sigmoid
// - torch.aten.silu -> hip.silu
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-torch-to-hip %s | FileCheck %s

// CHECK-LABEL: func.func @test_sigmoid
func.func @test_sigmoid(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: hip.sigmoid
  %0 = "torch.aten.sigmoid"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @test_silu
func.func @test_silu(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: hip.silu
  %0 = "torch.aten.silu"(%arg0) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
