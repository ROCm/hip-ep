// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// Constant parameters plus a static sliced-axis bound give an exact extent.
// The untouched dynamic axis passes through directly from data.
// CHECK-LABEL: func.func @constant_negative_step_dynamic_untouched
// CHECK-SAME: %[[DATA:[^,]+]]: tensor<6x?xf32>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[DATA]], %[[C1]]
// CHECK-NOT: hip.readback
// CHECK: return %[[C3]], %[[D1]] : index, index
func.func @constant_negative_step_dynamic_untouched(
    %ctx: !hip.context,
    %data: tensor<6x?xf32>,
    %init: tensor<?x?xf32>) -> (index, index) {
  %starts = arith.constant dense<[5]> : tensor<1xi64>
  %ends = arith.constant dense<[0]> : tensor<1xi64>
  %axes = arith.constant dense<[0]> : tensor<1xi64>
  %steps = arith.constant dense<[-2]> : tensor<1xi64>
  %result = hip.slice(%ctx)
    ins(%data, %starts, %ends :
        tensor<6x?xf32>, tensor<1xi64>, tensor<1xi64>)
    axes(%axes : tensor<1xi64>)
    steps(%steps : tensor<1xi64>)
    outs(%init : tensor<?x?xf32>)
    : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}

// -----

// A dynamic sliced-axis bound needs runtime clamping. The native Slice
// destination is physical capacity, so reification deliberately lifts outs
// rather than claiming that data's capacity is the logical slice extent.
// CHECK-LABEL: func.func @dynamic_sliced_axis_lifts_capacity
// CHECK-SAME: %[[DATA:[^,]+]]: tensor<?xf32>
// CHECK-SAME: %[[INIT:[^)]+]]: tensor<?xf32>
// CHECK-NOT: tensor.dim %[[DATA]]
// CHECK-NOT: hip.readback
// CHECK: %[[CAPACITY:.*]] = tensor.dim %[[INIT]]
// CHECK: return %[[CAPACITY]] : index
func.func @dynamic_sliced_axis_lifts_capacity(
    %ctx: !hip.context,
    %data: tensor<?xf32>,
    %init: tensor<?xf32>) -> index {
  %starts = arith.constant dense<[1]> : tensor<1xi64>
  %ends = arith.constant dense<[3]> : tensor<1xi64>
  %axes = arith.constant dense<[0]> : tensor<1xi64>
  %steps = arith.constant dense<[1]> : tensor<1xi64>
  %result = hip.slice(%ctx)
    ins(%data, %starts, %ends :
        tensor<?xf32>, tensor<1xi64>, tensor<1xi64>)
    axes(%axes : tensor<1xi64>)
    steps(%steps : tensor<1xi64>)
    outs(%init : tensor<?xf32>)
    : tensor<?xf32>
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %result, %c0 : tensor<?xf32>
  return %d0 : index
}
