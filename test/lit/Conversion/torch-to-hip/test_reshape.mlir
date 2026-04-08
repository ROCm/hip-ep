// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify Torch unsqueeze is correctly lowered to tensor.expand_shape.
//
// This test validates:
// - torch.aten.unsqueeze -> tensor.expand_shape
// - Proper handling of the dim argument
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-torch-to-hip %s | FileCheck %s

// CHECK-LABEL: func.func @test_unsqueeze
func.func @test_unsqueeze(%arg0: tensor<3x4xf32>) -> tensor<3x1x4xf32> {
  %int1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
  // CHECK: tensor.expand_shape
  %0 = "torch.aten.unsqueeze"(%arg0, %int1) : (tensor<3x4xf32>, !torch.int) -> tensor<3x1x4xf32>
  return %0 : tensor<3x1x4xf32>
}
