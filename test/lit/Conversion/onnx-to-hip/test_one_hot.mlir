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
}
