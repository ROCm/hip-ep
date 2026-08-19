// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --test-hip-whole-shape-dim-reify %s | FileCheck %s

// CHECK-LABEL: func.func @reify_dynamic_spatial
// CHECK: %[[N:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x4x?x?xf32>
// CHECK: %[[H:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x4x?x?xf32>
// CHECK: %[[HWIDE:.*]] = arith.index_cast %[[H]] : index to i128
// CHECK: %[[HQUOT:.*]] = arith.floordivsi %{{.*}}, %{{.*}} : i128
// CHECK: %[[HSAFE:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i128
// CHECK: %[[HI64:.*]] = arith.trunci %[[HSAFE]] : i128 to i64
// CHECK: %[[HOUT:.*]] = arith.index_cast %[[HI64]] : i64 to index
// CHECK: %[[W:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x4x?x?xf32>
// CHECK: %[[WWIDE:.*]] = arith.index_cast %[[W]] : index to i128
// CHECK: %[[WQUOT:.*]] = arith.floordivsi %{{.*}}, %{{.*}} : i128
// CHECK: %[[WSAFE:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i128
// CHECK: %[[WI64:.*]] = arith.trunci %[[WSAFE]] : i128 to i64
// CHECK: %[[WOUT:.*]] = arith.index_cast %[[WI64]] : i64 to index
// CHECK: return %[[N]], %[[HOUT]], %[[WOUT]] : index, index, index
func.func @reify_dynamic_spatial(
    %ctx: !hip.context,
    %valid: i1,
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
