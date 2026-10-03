// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --test-hip-whole-shape-dim-reify %s | FileCheck %s

// Proves that ConvTranspose reification derives dynamic result dimensions from
// the operands and attributes rather than echoing the DPS init.

// CHECK-LABEL: func.func @reify_dynamic_spatial
// CHECK-SAME: (%{{.*}}: !hip.context, %[[INPUT:.*]]: tensor<?x1x?x3xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[N:.*]] = tensor.dim %[[INPUT]], %[[C0]]
// CHECK: %[[H:.*]] = tensor.dim %[[INPUT]], %[[C2]]
// CHECK: %[[H2:.*]] = arith.muli %[[H]], %[[C2]]
// CHECK: %[[HOUT:.*]] = arith.addi %[[H2]], %[[C1]]
// CHECK: return %[[N]], %[[HOUT]] : index, index
func.func @reify_dynamic_spatial(
    %ctx: !hip.context,
    %input: tensor<?x1x?x3xf32>,
    %weights: tensor<1x2x3x3xf32>,
    %init: tensor<?x2x?x7xf32>) -> (index, index) {
  %result = hip.conv_transpose(%ctx)
    ins(%input, %weights : tensor<?x1x?x3xf32>, tensor<1x2x3x3xf32>)
    outs(%init : tensor<?x2x?x7xf32>)
    {kernel_shape = [3, 3], strides = [2, 2],
     pads = [0, 0, 0, 0], dilations = [1, 1],
     output_padding = [0, 0], group = 1}
    : tensor<?x2x?x7xf32>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %n = tensor.dim %result, %c0 : tensor<?x2x?x7xf32>
  %h = tensor.dim %result, %c2 : tensor<?x2x?x7xf32>
  return %n, %h : index, index
}

// CHECK-LABEL: func.func @reify_representable_offset_boundary
// CHECK-SAME: (%{{[^:]+}}: !hip.context,
// CHECK-SAME: %[[INPUT:[^:]+]]: tensor<1x1x?x1xf32>
// CHECK-DAG: %[[H:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK-DAG: %[[OFFSET:.*]] = arith.constant 9223372036854775806 : index
// CHECK: %[[HOUT:.*]] = arith.addi %[[H]], %[[OFFSET]]
// CHECK: return %[[HOUT]] : index
func.func @reify_representable_offset_boundary(
    %ctx: !hip.context,
    %input: tensor<1x1x?x1xf32>,
    %weights: tensor<1x1x1x1xf32>,
    %init: tensor<1x1x?x1xf32>) -> index {
  %result = hip.conv_transpose(%ctx)
    ins(%input, %weights : tensor<1x1x?x1xf32>, tensor<1x1x1x1xf32>)
    outs(%init : tensor<1x1x?x1xf32>)
    {kernel_shape = [9223372036854775807, 1], strides = [1, 1],
     pads = [0, 0, 0, 0], dilations = [1, 1],
     output_padding = [0, 0], group = 1}
    : tensor<1x1x?x1xf32>
  %c2 = arith.constant 2 : index
  %h = tensor.dim %result, %c2 : tensor<1x1x?x1xf32>
  return %h : index
}
