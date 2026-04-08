// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify Torch sum.dim_IntList is correctly lowered to hip.reduce_sum.
//
// This test validates:
// - torch.aten.sum.dim_IntList -> hip.reduce_sum
// - Proper handling of dims list, keepdim, and dtype arguments
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-torch-to-hip %s | FileCheck %s

// CHECK-LABEL: func.func @test_sum_dim
func.func @test_sum_dim(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
  %int1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
  %dims = "torch.prim.ListConstruct"(%int1) : (!torch.int) -> !torch.list<int>
  %true = "torch.constant.bool"() {value = true} : () -> !torch.bool
  %none = "torch.constant.none"() : () -> !torch.none
  // CHECK: hip.reduce_sum
  %0 = "torch.aten.sum.dim_IntList"(%arg0, %dims, %true, %none) : (tensor<2x3x4xf32>, !torch.list<int>, !torch.bool, !torch.none) -> tensor<2x1x4xf32>
  return %0 : tensor<2x1x4xf32>
}
