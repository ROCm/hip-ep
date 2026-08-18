// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Unit test verifying hip.qmatmul operation parses correctly.
// Demonstrates PDLL-style QDQ fusion target operation.
// ============================================================================

// RUN: hip-mlir-opt %s | FileCheck %s

module {
  func.func @test(%ctx: !hip.context, %lhs: tensor<4x128xf32>, %rhs: tensor<128x256xf32>) -> tensor<4x256xf32> {
    %output = tensor.empty() : tensor<4x256xf32>
    %result = hip.qmatmul(%ctx) ins(%lhs, %rhs : tensor<4x128xf32>, tensor<128x256xf32>)
                           outs(%output : tensor<4x256xf32>)
                           {lhs_scale = 0.1 : f32, rhs_scale = 0.05 : f32, output_scale = 0.2 : f32}
                           : tensor<4x256xf32>
    return %result : tensor<4x256xf32>
  }
}

// CHECK-LABEL: func.func @test
// CHECK: hip.qmatmul
// CHECK-SAME: {lhs_scale = 1.000000e-01 : f32, output_scale = 2.000000e-01 : f32, rhs_scale = 5.000000e-02 : f32}
