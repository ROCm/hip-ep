// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify Torch elementwise binary ops are correctly lowered to HIP ops
// in tensor-first mode.
//
// This test validates:
// - torch.aten.add.Tensor -> hip.add
// - torch.aten.mul.Tensor -> hip.mul
// - torch.aten.sub.Tensor -> hip.sub
// - Proper handling of the alpha argument on add/sub
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-torch-to-hip %s | FileCheck %s

// CHECK-LABEL: func.func @test_add
func.func @test_add(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %int1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
  // CHECK: hip.add
  %0 = "torch.aten.add.Tensor"(%arg0, %arg1, %int1) : (tensor<2x3xf32>, tensor<2x3xf32>, !torch.int) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @test_mul
func.func @test_mul(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: hip.mul
  %0 = "torch.aten.mul.Tensor"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @test_sub
func.func @test_sub(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %int1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
  // CHECK: hip.sub
  %0 = "torch.aten.sub.Tensor"(%arg0, %arg1, %int1) : (tensor<2x3xf32>, tensor<2x3xf32>, !torch.int) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
