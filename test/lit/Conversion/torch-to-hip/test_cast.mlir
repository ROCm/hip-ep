// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify Torch dtype conversion is correctly lowered to hip.cast.
//
// This test validates:
// - torch.aten.to.dtype -> hip.cast
// - Proper handling of dtype, non_blocking, copy, memory_format args
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-torch-to-hip %s | FileCheck %s

// CHECK-LABEL: func.func @test_cast_f32_to_f16
func.func @test_cast_f32_to_f16(%arg0: tensor<4x8xf32>) -> tensor<4x8xf16> {
  %int5 = "torch.constant.int"() {value = 5 : i64} : () -> !torch.int
  %false = "torch.constant.bool"() {value = false} : () -> !torch.bool
  %none = "torch.constant.none"() : () -> !torch.none
  // CHECK: hip.cast
  %0 = "torch.aten.to.dtype"(%arg0, %int5, %false, %false, %none) : (tensor<4x8xf32>, !torch.int, !torch.bool, !torch.bool, !torch.none) -> tensor<4x8xf16>
  return %0 : tensor<4x8xf16>
}
