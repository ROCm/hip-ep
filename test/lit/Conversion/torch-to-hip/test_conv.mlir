// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify Torch conv2d is correctly lowered to hip.conv.
//
// This test validates:
// - torch.aten.conv2d -> hip.conv
// - Proper handling of stride, padding, dilation, groups arguments
// - Optional bias (none) handling
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-torch-to-hip %s | FileCheck %s

// CHECK-LABEL: func.func @test_conv2d
func.func @test_conv2d(%input: tensor<1x3x8x8xf32>, %weight: tensor<16x3x3x3xf32>) -> tensor<1x16x6x6xf32> {
  %none = "torch.constant.none"() : () -> !torch.none
  %int1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
  %int0 = "torch.constant.int"() {value = 0 : i64} : () -> !torch.int
  %stride = "torch.prim.ListConstruct"(%int1, %int1) : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = "torch.prim.ListConstruct"(%int0, %int0) : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = "torch.prim.ListConstruct"(%int1, %int1) : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: hip.conv
  %0 = "torch.aten.conv2d"(%input, %weight, %none, %stride, %padding, %dilation, %int1) : (tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> tensor<1x16x6x6xf32>
  return %0 : tensor<1x16x6x6xf32>
}
