// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// CHECK-LABEL: func.func @reify_floor_with_indices
// CHECK: %[[N:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x3x?xf32>
// CHECK: %[[L:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x3x?xf32>
// CHECK: %[[WIDE:.*]] = arith.index_cast %[[L]] : index to i128
// CHECK: %[[QUOT:.*]] = arith.floordivsi %{{.*}}, %{{.*}} : i128
// CHECK: %[[RAW:.*]] = arith.addi %[[QUOT]], %{{.*}} : i128
// CHECK: %[[SAFE:.*]] = arith.select %{{.*}}, %[[RAW]], %{{.*}} : i128
// CHECK: %[[I64:.*]] = arith.trunci %[[SAFE]] : i128 to i64
// CHECK: %[[VOUT:.*]] = arith.index_cast %[[I64]] : i64 to index
// CHECK: %[[IL:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x3x?xf32>
// CHECK: %[[IWIDE:.*]] = arith.index_cast %[[IL]] : index to i128
// CHECK: %[[IQUOT:.*]] = arith.floordivsi %{{.*}}, %{{.*}} : i128
// CHECK: %[[ISAFE:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i128
// CHECK: %[[II64:.*]] = arith.trunci %[[ISAFE]] : i128 to i64
// CHECK: %[[IOUT:.*]] = arith.index_cast %[[II64]] : i64 to index
// CHECK: return %[[N]], %[[VOUT]], %[[IOUT]] : index, index, index
func.func @reify_floor_with_indices(
    %ctx: !hip.context,
    %valid: i1,
    %input: tensor<?x3x?xf32>,
    %values_init: tensor<?x3x?xf32>,
    %indices_init: tensor<?x3x?xi64>) -> (index, index, index) {
  %values, %indices = hip.pool(%ctx) valid(%valid)
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

// CHECK-LABEL: func.func @reify_ceil_trailing_window
// CHECK: %[[L:.*]] = tensor.dim
// CHECK: %[[WIDE:.*]] = arith.index_cast %[[L]] : index to i128
// CHECK: %[[NUM:.*]] = arith.subi %{{.*}}, %{{.*}} : i128
// CHECK: %[[QUOT:.*]] = arith.ceildivsi %[[NUM]], %{{.*}} : i128
// CHECK: %[[RAW:.*]] = arith.addi %[[QUOT]], %{{.*}} : i128
// CHECK: %[[POSITIVE:.*]] = arith.cmpi sgt, %[[RAW]], %{{.*}} : i128
// CHECK: %[[START:.*]] = arith.muli %[[QUOT]], %{{.*}} : i128
// CHECK: %[[LIMIT:.*]] = arith.addi %[[WIDE]], %{{.*}} : i128
// CHECK: %[[TRAILING:.*]] = arith.cmpi sge, %[[START]], %[[LIMIT]] : i128
// CHECK: %[[SHOULD_DECREMENT:.*]] = arith.andi %[[POSITIVE]], %[[TRAILING]]
// CHECK: %[[CORRECTED:.*]] = arith.select %[[SHOULD_DECREMENT]], %[[QUOT]], %[[RAW]] : i128
// CHECK: %[[NONNEGATIVE:.*]] = arith.cmpi sge, %[[CORRECTED]], %{{.*}} : i128
// CHECK: %[[IN_RANGE:.*]] = arith.cmpi sle, %[[CORRECTED]], %{{.*}} : i128
// CHECK: %[[VALID:.*]] = arith.andi %[[NONNEGATIVE]], %[[IN_RANGE]]
// CHECK: %[[SAFE:.*]] = arith.select %[[VALID]], %[[CORRECTED]], %{{.*}} : i128
// CHECK: %[[I64:.*]] = arith.trunci %[[SAFE]] : i128 to i64
// CHECK: %[[OUT:.*]] = arith.index_cast %[[I64]] : i64 to index
// CHECK: return %[[OUT]] : index
func.func @reify_ceil_trailing_window(
    %ctx: !hip.context,
    %valid: i1,
    %input: tensor<1x3x?xf32>,
    %init: tensor<1x3x?xf32>) -> index {
  %result = hip.pool(%ctx) valid(%valid)
    ins(%input : tensor<1x3x?xf32>)
    outs(%init : tensor<1x3x?xf32>)
    {pool_mode = 0, kernel_shape = [4], strides = [3], pads = [1, 0],
     dilations = [1], ceil_mode = 1, storage_order = 0}
    : tensor<1x3x?xf32>
  %c2 = arith.constant 2 : index
  %l = tensor.dim %result, %c2 : tensor<1x3x?xf32>
  return %l : index
}
