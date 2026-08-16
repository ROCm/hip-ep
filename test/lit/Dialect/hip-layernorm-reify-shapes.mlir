// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// CHECK-LABEL: func.func @dynamic_stats
// CHECK-SAME: %[[INPUT:[^,]+]]: tensor<?x3x4xf16>
// CHECK-DAG: %[[YD:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK-DAG: %[[MD:.*]] = tensor.dim %[[INPUT]], %{{.*}}
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: return %[[YD]], %[[MD]], %[[ONE]] : index, index, index
func.func @dynamic_stats(
    %ctx: !hip.context,
    %input: tensor<?x3x4xf16>,
    %scale: tensor<3x4xf16>,
    %y_init: tensor<?x3x4xf16>,
    %mean_init: tensor<?x1x1xf32>,
    %inv_init: tensor<?x1x1xf32>) -> (index, index, index) {
  %result:3 = hip.layer_norm(%ctx)
    ins(%input, %scale : tensor<?x3x4xf16>, tensor<3x4xf16>)
    outs(%y_init, %mean_init, %inv_init :
         tensor<?x3x4xf16>, tensor<?x1x1xf32>, tensor<?x1x1xf32>)
    {axis = 1 : i64, stash_type = 1 : i64}
    : tensor<?x3x4xf16>, tensor<?x1x1xf32>, tensor<?x1x1xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %yd = tensor.dim %result#0, %c0 : tensor<?x3x4xf16>
  %md = tensor.dim %result#1, %c0 : tensor<?x1x1xf32>
  %one = tensor.dim %result#2, %c1 : tensor<?x1x1xf32>
  return %yd, %md, %one : index, index, index
}
