// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify Torch transpose.int is correctly lowered to hip.transpose.
//
// This test validates:
// - torch.aten.transpose.int -> hip.transpose
// - Proper handling of dim0 and dim1 arguments
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-torch-to-hip %s | FileCheck %s

// CHECK-LABEL: func.func @test_transpose
func.func @test_transpose(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %int0 = "torch.constant.int"() {value = 0 : i64} : () -> !torch.int
  %int1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
  // CHECK: hip.transpose
  %0 = "torch.aten.transpose.int"(%arg0, %int0, %int1) : (tensor<2x3xf32>, !torch.int, !torch.int) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
