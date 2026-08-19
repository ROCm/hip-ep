// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --test-hip-whole-shape-dim-reify %s | FileCheck %s

// A static tuple width maps the dynamic outer-indices and data-tail dimensions
// directly to their authoritative operands.
// CHECK-LABEL: func.func @static_tuple_width
// CHECK-SAME: %[[DATA:[^,]+]]: tensor<4x?xf32>
// CHECK-SAME: %[[INDICES:[^,]+]]: tensor<?x1xi64>
// CHECK-DAG: %[[OUTER:.*]] = tensor.dim %[[INDICES]], %{{.*}}
// CHECK-DAG: %[[TAIL:.*]] = tensor.dim %[[DATA]], %{{.*}}
// CHECK: return %[[OUTER]], %[[TAIL]] : index, index
func.func @static_tuple_width(
    %ctx: !hip.context,
    %data: tensor<4x?xf32>,
    %indices: tensor<?x1xi64>,
    %output: tensor<?x?xf32>) -> (index, index) {
  %result = hip.gather_nd(%ctx)
      ins(%data, %indices : tensor<4x?xf32>, tensor<?x1xi64>)
      outs(%output : tensor<?x?xf32>)
      : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %outer = tensor.dim %result, %c0 : tensor<?x?xf32>
  %tail = tensor.dim %result, %c1 : tensor<?x?xf32>
  return %outer, %tail : index, index
}

// A dynamic tuple width keeps the existing outs-authoritative fallback.
// CHECK-LABEL: func.func @dynamic_tuple_width
// CHECK-SAME: %[[OUTPUT:[^,)]+]]: tensor<?xf32>
// CHECK: %[[DIM:.*]] = tensor.dim %[[OUTPUT]], %{{.*}}
// CHECK: return %[[DIM]] : index
func.func @dynamic_tuple_width(
    %ctx: !hip.context,
    %data: tensor<2x2xf32>,
    %indices: tensor<2x?xi64>,
    %output: tensor<?xf32>) -> index {
  %result = hip.gather_nd(%ctx)
      ins(%data, %indices : tensor<2x2xf32>, tensor<2x?xi64>)
      outs(%output : tensor<?xf32>)
      : tensor<?xf32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %result, %c0 : tensor<?xf32>
  return %dim : index
}
