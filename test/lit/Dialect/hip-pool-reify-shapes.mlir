// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// CHECK-LABEL: func.func @reify_floor_with_indices
// CHECK: %[[N:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x3x?xf32>
// CHECK: %[[L:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x3x?xf32>
// CHECK: %[[NUM:.*]] = arith.addi %[[L]],
// CHECK: %[[QUOT:.*]] = arith.floordivsi %[[NUM]],
// CHECK: %[[VOUT:.*]] = arith.addi %[[QUOT]],
// CHECK: %[[IL:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x3x?xf32>
// CHECK: %[[INUM:.*]] = arith.addi %[[IL]],
// CHECK: %[[IQUOT:.*]] = arith.floordivsi %[[INUM]],
// CHECK: %[[IOUT:.*]] = arith.addi %[[IQUOT]],
// CHECK: return %[[N]], %[[VOUT]], %[[IOUT]] : index, index, index
func.func @reify_floor_with_indices(
    %ctx: !hip.context,
    %input: tensor<?x3x?xf32>,
    %values_init: tensor<?x3x?xf32>,
    %indices_init: tensor<?x3x?xi64>) -> (index, index, index) {
  %values, %indices = hip.pool(%ctx)
    ins(%input : tensor<?x3x?xf32>)
    outs(%values_init, %indices_init : tensor<?x3x?xf32>, tensor<?x3x?xi64>)
    {pool_mode = 1, kernel_shape = [3], strides = [2], pads = [1, 1],
     dilations = [1], ceil_mode = 0, storage_order = 0}
    : tensor<?x3x?xf32>, tensor<?x3x?xi64>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %n = tensor.dim %values, %c0 : tensor<?x3x?xf32>
  %vl = tensor.dim %values, %c2 : tensor<?x3x?xf32>
  %il = tensor.dim %indices, %c2 : tensor<?x3x?xi64>
  return %n, %vl, %il : index, index, index
}

// CHECK-LABEL: func.func @reify_ceil
// CHECK: %[[L:.*]] = tensor.dim
// CHECK: %[[NUM:.*]] = arith.addi %[[L]],
// CHECK: %[[QUOT:.*]] = arith.ceildivsi %[[NUM]],
// CHECK: %[[LOUT:.*]] = arith.addi %[[QUOT]],
// CHECK: return %[[LOUT]] : index
func.func @reify_ceil(
    %ctx: !hip.context,
    %input: tensor<1x3x?xf32>,
    %init: tensor<1x3x?xf32>) -> index {
  %result = hip.pool(%ctx)
    ins(%input : tensor<1x3x?xf32>)
    outs(%init : tensor<1x3x?xf32>)
    {pool_mode = 0, kernel_shape = [4], strides = [3], pads = [1, 0],
     dilations = [1], ceil_mode = 1, storage_order = 0}
    : tensor<1x3x?xf32>
  %c2 = arith.constant 2 : index
  %l = tensor.dim %result, %c2 : tensor<1x3x?xf32>
  return %l : index
}
