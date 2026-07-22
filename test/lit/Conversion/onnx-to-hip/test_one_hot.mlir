// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
    return %arg0 : tensor<2x2xi64>
  }

  func.func @onehot_axis1(
      %arg0: tensor<2x2xi64>,
      %arg1: tensor<i64>,
      %arg2: tensor<2xf32>) -> tensor<2x10x2xf32> {
    %result = "onnx.OneHot"(%arg0, %arg1, %arg2) {axis = 1 : si64}
        : (tensor<2x2xi64>, tensor<i64>, tensor<2xf32>) -> tensor<2x10x2xf32>
    return %result : tensor<2x10x2xf32>
  }

  // CHECK-LABEL: func.func @onehot_axis1
  // CHECK: hip.one_hot(%[[CTX:.*]]) ins(%[[IDX:.*]], %[[DEPTH:.*]], %[[VAL:.*]] : tensor<2x2xi64>, tensor<i64>, tensor<2xf32>) outs({{.*}} : tensor<2x10x2xf32>) {axis = 1 : i64} : tensor<2x10x2xf32>
  // CHECK-NOT: onnx.OneHot

  func.func @onehot_default_axis(
      %arg0: tensor<2xi64>,
      %arg1: tensor<i64>,
      %arg2: tensor<2xf32>) -> tensor<2x4xf32> {
    %result = "onnx.OneHot"(%arg0, %arg1, %arg2)
        : (tensor<2xi64>, tensor<i64>, tensor<2xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @onehot_default_axis
  // CHECK: hip.one_hot(%{{.*}}) ins({{.*}} : tensor<2xi64>, tensor<i64>, tensor<2xf32>) outs({{.*}} : tensor<2x4xf32>) : tensor<2x4xf32>

  // Dynamic depth axis: the axis extent is the runtime *value* of the depth
  // scalar (e.g. a pooler that computes the group count at runtime), so the
  // output axis dim is dynamic. The extent MUST be read back to host and used
  // to size tensor.empty -- NOT hardcoded to 1 (which would drop every
  // index >= 1 in the scatter and collapse the axis to a single row).
  func.func @onehot_dynamic_depth_scalar(
      %arg0: tensor<1x8xi64>,
      %arg1: tensor<i64>,
      %arg2: tensor<2xf32>) -> tensor<1x8x?xf32> {
    %result = "onnx.OneHot"(%arg0, %arg1, %arg2) {axis = 2 : si64}
        : (tensor<1x8xi64>, tensor<i64>, tensor<2xf32>) -> tensor<1x8x?xf32>
    return %result : tensor<1x8x?xf32>
  }

  // CHECK-LABEL: func.func @onehot_dynamic_depth_scalar
  // CHECK: %[[D:.*]] = hip.readback_scalar(%[[CTX:.*]], %{{.*}} : tensor<i64>) -> i64
  // CHECK: %[[DI:.*]] = arith.index_cast %[[D]] : i64 to index
  // CHECK: tensor.empty(%[[DI]]) : tensor<1x8x?xf32>
  // CHECK: hip.one_hot(%[[CTX]]) ins({{.*}} : tensor<1x8xi64>, tensor<i64>, tensor<2xf32>) outs({{.*}} : tensor<1x8x?xf32>) {axis = 2 : i64} : tensor<1x8x?xf32>
  // CHECK-NOT: arith.constant 1 : index
  // CHECK-NOT: onnx.OneHot

  // Single-element rank-1 depth (some exporters emit tensor<1xi64>) is
  // collapsed to rank-0 before the readback, matching the Range convention.
  func.func @onehot_dynamic_depth_rank1(
      %arg0: tensor<1x8xi64>,
      %arg1: tensor<1xi64>,
      %arg2: tensor<2xf32>) -> tensor<1x8x?xf32> {
    %result = "onnx.OneHot"(%arg0, %arg1, %arg2) {axis = 2 : si64}
        : (tensor<1x8xi64>, tensor<1xi64>, tensor<2xf32>) -> tensor<1x8x?xf32>
    return %result : tensor<1x8x?xf32>
  }

  // CHECK-LABEL: func.func @onehot_dynamic_depth_rank1
  // CHECK: %[[C:.*]] = tensor.collapse_shape %{{.*}} [] : tensor<1xi64> into tensor<i64>
  // CHECK: %[[D2:.*]] = hip.readback_scalar(%{{.*}}, %[[C]] : tensor<i64>) -> i64
  // CHECK: arith.index_cast %[[D2]] : i64 to index
  // CHECK-NOT: onnx.OneHot
}
