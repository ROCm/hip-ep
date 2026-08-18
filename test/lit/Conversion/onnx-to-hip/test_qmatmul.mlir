// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.qmatmul operation is properly defined and can be parsed/verified.
// This demonstrates the quantized matmul operation with integrated QDQ scales.
//
// NOTE: This is a demonstrative test showing the operation definition.
// The fusion pattern (QuantizeLinear -> MatMul -> DequantizeLinear => qmatmul)
// would require additional pattern infrastructure.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg %s | FileCheck %s

module {
  // Test basic qmatmul operation with static shapes
  func.func @qmatmul_static(%ctx: !hip.context,
                             %lhs: tensor<4x128xf32>,
                             %rhs: tensor<128x256xf32>) -> tensor<4x256xf32> {
    %output = tensor.empty() : tensor<4x256xf32>
    %result = hip.qmatmul(%ctx) ins(%lhs, %rhs : tensor<4x128xf32>, tensor<128x256xf32>)
                           outs(%output : tensor<4x256xf32>)
                           {lhs_scale = 0.1 : f32, rhs_scale = 0.05 : f32, output_scale = 0.2 : f32}
                           : tensor<4x256xf32>
    return %result : tensor<4x256xf32>
  }

  // Test qmatmul with dynamic shapes
  func.func @qmatmul_dynamic(%ctx: !hip.context,
                              %lhs: tensor<?x?xf32>,
                              %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %d1 = tensor.dim %rhs, %c1 : tensor<?x?xf32>
    %output = tensor.empty(%d0, %d1) : tensor<?x?xf32>
    %result = hip.qmatmul(%ctx) ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
                           outs(%output : tensor<?x?xf32>)
                           {lhs_scale = 0.15 : f32, rhs_scale = 0.08 : f32, output_scale = 0.25 : f32}
                           : tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }

  // Test qmatmul with batch dimensions
  func.func @qmatmul_batch(%ctx: !hip.context,
                            %lhs: tensor<2x4x128xf32>,
                            %rhs: tensor<2x128x256xf32>) -> tensor<2x4x256xf32> {
    %output = tensor.empty() : tensor<2x4x256xf32>
    %result = hip.qmatmul(%ctx) ins(%lhs, %rhs : tensor<2x4x128xf32>, tensor<2x128x256xf32>)
                           outs(%output : tensor<2x4x256xf32>)
                           {lhs_scale = 0.12 : f32, rhs_scale = 0.06 : f32, output_scale = 0.18 : f32}
                           : tensor<2x4x256xf32>
    return %result : tensor<2x4x256xf32>
  }
}

// Verify the operation parses correctly and preserves attributes
// CHECK-LABEL: func.func @qmatmul_static
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<4x128xf32>, %[[RHS:.*]]: tensor<128x256xf32>)
// CHECK: %[[OUTPUT:.*]] = tensor.empty() : tensor<4x256xf32>
// CHECK: %[[RESULT:.*]] = hip.qmatmul(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<4x128xf32>, tensor<128x256xf32>)
// CHECK-SAME: outs(%[[OUTPUT]] : tensor<4x256xf32>)
// CHECK-SAME: {lhs_scale = 1.000000e-01 : f32, output_scale = 2.000000e-01 : f32, rhs_scale = 5.000000e-02 : f32}
// CHECK-SAME: : tensor<4x256xf32>

// CHECK-LABEL: func.func @qmatmul_dynamic
// CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[LHS2:.*]]: tensor<?x?xf32>, %[[RHS2:.*]]: tensor<?x?xf32>)
// CHECK: hip.qmatmul(%[[CTX2]]) ins(%[[LHS2]], %[[RHS2]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: {lhs_scale = 1.500000e-01 : f32, output_scale = 2.500000e-01 : f32, rhs_scale = 8.000000e-02 : f32}

// CHECK-LABEL: func.func @qmatmul_batch
// CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[LHS3:.*]]: tensor<2x4x128xf32>, %[[RHS3:.*]]: tensor<2x128x256xf32>)
// CHECK: hip.qmatmul(%[[CTX3]]) ins(%[[LHS3]], %[[RHS3]] : tensor<2x4x128xf32>, tensor<2x128x256xf32>)
// CHECK-SAME: {lhs_scale = 1.200000e-01 : f32, output_scale = 1.800000e-01 : f32, rhs_scale = 6.000000e-02 : f32}
