// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// Stamped pads/axes are compile-time shape metadata even when the payload
// operands are function arguments. Dynamic input extents use the exact affine
// rule; reification does not read the pads payload.
// CHECK-LABEL: func.func @stamped_pads_dynamic_input
// CHECK-SAME: %[[DATA:[^,]+]]: tensor<?x?xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[DATA]], %[[C0]]
// CHECK: %[[D1:.*]] = tensor.dim %[[DATA]], %[[C1]]
// CHECK: %[[PADDED:.*]] = arith.addi %[[D1]], %[[C3]] : index
// CHECK-NOT: hip.readback
// CHECK: return %[[D0]], %[[PADDED]] : index, index
func.func @stamped_pads_dynamic_input(
    %ctx: !hip.context,
    %data: tensor<?x?xf32>,
    %pads: tensor<2xi64>,
    %axes: tensor<1xi64>,
    %init: tensor<?x?xf32>) -> (index, index) {
  %result = hip.pad(%ctx)
    ins(%data, %pads : tensor<?x?xf32>, tensor<2xi64>)
    axes(%axes : tensor<1xi64>)
    outs(%init : tensor<?x?xf32>)
    {static_pads = array<i64: 1, 2>, static_axes = array<i64: 1>}
    : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}

// -----

// Runtime pads are intentionally not read by dialect reification. The
// converter has already sized the destination (using synchronized readback),
// so the honest fallback is its mixed shape.
// CHECK-LABEL: func.func @runtime_pads_lift_destination
// CHECK-SAME: %[[DATA:[^,]+]]: tensor<?x?xf32>
// CHECK-SAME: %[[PADS:[^,]+]]: tensor<4xi64>
// CHECK-SAME: %[[INIT:[^)]+]]: tensor<?x?xf32>
// CHECK-NOT: tensor.dim %[[DATA]]
// CHECK-NOT: hip.readback
// CHECK: tensor.dim %[[INIT]]
// CHECK: tensor.dim %[[INIT]]
func.func @runtime_pads_lift_destination(
    %ctx: !hip.context,
    %data: tensor<?x?xf32>,
    %pads: tensor<4xi64>,
    %init: tensor<?x?xf32>) -> (index, index) {
  %result = hip.pad(%ctx)
    ins(%data, %pads : tensor<?x?xf32>, tensor<4xi64>)
    outs(%init : tensor<?x?xf32>)
    : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}

// -----

// An unrepresentable static pad offset fails Tier-1 reification before reading
// data dimensions and falls back to the destination shape.
// CHECK-LABEL: func.func @overflowing_static_pad_offset
// CHECK-SAME: (%{{[^:]+}}: !hip.context, %[[DATA:[^:]+]]: tensor<?xf32>,
// CHECK-SAME: %{{[^:]+}}: tensor<2xi64>, %[[INIT:[^:]+]]: tensor<?xf32>)
// CHECK-NOT: tensor.dim %[[DATA]]
// CHECK: tensor.dim %[[INIT]]
func.func @overflowing_static_pad_offset(
    %ctx: !hip.context,
    %data: tensor<?xf32>,
    %pads: tensor<2xi64>,
    %init: tensor<?xf32>) -> index {
  %result = hip.pad(%ctx)
    ins(%data, %pads : tensor<?xf32>, tensor<2xi64>)
    outs(%init : tensor<?xf32>)
    {static_pads = array<i64: 9223372036854775807, 9223372036854775807>}
    : tensor<?xf32>
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %result, %c0 : tensor<?xf32>
  return %d0 : index
}

// -----

// CHECK-LABEL: func.func @representable_static_pad_boundary
// CHECK: %[[MAX:.*]] = arith.constant 9223372036854775807 : index
// CHECK: return %[[MAX]] : index
func.func @representable_static_pad_boundary(
    %ctx: !hip.context,
    %data: tensor<0xf32>,
    %pads: tensor<2xi64>,
    %init: tensor<?xf32>) -> index {
  %result = hip.pad(%ctx)
    ins(%data, %pads : tensor<0xf32>, tensor<2xi64>)
    outs(%init : tensor<?xf32>)
    {static_pads = array<i64: 9223372036854775807, 0>}
    : tensor<?xf32>
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %result, %c0 : tensor<?xf32>
  return %d0 : index
}
