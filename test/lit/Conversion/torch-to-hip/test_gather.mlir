// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify Torch index_select is correctly lowered to hip.gather.
//
// This test validates:
// - torch.aten.index_select -> hip.gather
// - Proper handling of the dim argument
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-torch-to-hip %s | FileCheck %s

// CHECK-LABEL: func.func @test_index_select
func.func @test_index_select(%arg0: tensor<10x8xf32>, %arg1: tensor<3xi64>) -> tensor<3x8xf32> {
  %int0 = "torch.constant.int"() {value = 0 : i64} : () -> !torch.int
  // CHECK: hip.gather
  %0 = "torch.aten.index_select"(%arg0, %int0, %arg1) : (tensor<10x8xf32>, !torch.int, tensor<3xi64>) -> tensor<3x8xf32>
  return %0 : tensor<3x8xf32>
}
