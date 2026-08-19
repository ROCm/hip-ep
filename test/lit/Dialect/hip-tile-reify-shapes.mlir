// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --test-hip-whole-shape-dim-reify %s | FileCheck %s

// CHECK-LABEL: func.func @dynamic_input_constant_repeats
// CHECK-SAME: (%{{.*}}: !hip.context, %[[INPUT:[^,]+]]: tensor<?x?xf32>,
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[INPUT]], %[[C0]]
// CHECK: %[[O0:.*]] = arith.muli %[[D0]], %[[C2]]
// CHECK: %[[D1:.*]] = tensor.dim %[[INPUT]], %[[C1]]
// CHECK: %[[O1:.*]] = arith.muli %[[D1]], %[[C3]]
// CHECK: return %[[O0]], %[[O1]] : index, index
func.func @dynamic_input_constant_repeats(
    %ctx: !hip.context,
    %input: tensor<?x?xf32>,
    %repeats: tensor<2xi64>,
    %init: tensor<?x?xf32>) -> (index, index) {
  %result = hip.tile(%ctx)
    ins(%input, %repeats : tensor<?x?xf32>, tensor<2xi64>)
    outs(%init : tensor<?x?xf32>)
    {static_repeats = array<i64: 2, 3>}
    : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}

// -----

// CHECK-LABEL: func.func @static_product_i64_boundary
// CHECK: %[[MAX:.*]] = arith.constant 9223372036854775807 : index
// CHECK: return %[[MAX]] : index
func.func @static_product_i64_boundary(
    %ctx: !hip.context,
    %input: tensor<1xf32>,
    %repeats: tensor<1xi64>,
    %init: tensor<9223372036854775807xf32>) -> index {
  %result = hip.tile(%ctx)
    ins(%input, %repeats : tensor<1xf32>, tensor<1xi64>)
    outs(%init : tensor<9223372036854775807xf32>)
    {static_repeats = array<i64: 9223372036854775807>}
    : tensor<9223372036854775807xf32>
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %result, %c0 : tensor<9223372036854775807xf32>
  return %d0 : index
}
