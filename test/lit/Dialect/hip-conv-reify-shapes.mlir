// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// CHECK-LABEL: func.func @reify_dynamic_spatial
// CHECK: %[[N:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x4x?x?xf32>
// CHECK: %[[H:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x4x?x?xf32>
// CHECK: %[[HNUM:.*]] = arith.addi %[[H]],
// CHECK: %[[HQUOT:.*]] = arith.floordivsi %[[HNUM]],
// CHECK: %[[HOUT:.*]] = arith.addi %[[HQUOT]],
// CHECK: %[[W:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x4x?x?xf32>
// CHECK: %[[WNUM:.*]] = arith.addi %[[W]],
// CHECK: %[[WQUOT:.*]] = arith.floordivsi %[[WNUM]],
// CHECK: %[[WOUT:.*]] = arith.addi %[[WQUOT]],
// CHECK: return %[[N]], %[[HOUT]], %[[WOUT]] : index, index, index
func.func @reify_dynamic_spatial(
    %ctx: !hip.context,
    %input: tensor<?x4x?x?xf32>,
    %weights: tensor<8x2x3x5xf32>,
    %init: tensor<?x8x?x?xf32>) -> (index, index, index) {
  %result = hip.conv(%ctx)
    ins(%input, %weights : tensor<?x4x?x?xf32>, tensor<8x2x3x5xf32>)
    outs(%init : tensor<?x8x?x?xf32>)
    {kernel_shape = [3, 5], strides = [2, 3],
     pads = [1, 2, 1, 2], dilations = [1, 1], group = 2}
    : tensor<?x8x?x?xf32>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %n = tensor.dim %result, %c0 : tensor<?x8x?x?xf32>
  %h = tensor.dim %result, %c2 : tensor<?x8x?x?xf32>
  %w = tensor.dim %result, %c3 : tensor<?x8x?x?xf32>
  return %n, %h, %w : index, index, index
}
