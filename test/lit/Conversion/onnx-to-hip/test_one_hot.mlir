// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @onehot_axis1(
      %arg0: tensor<2x2xi64> {onnx.name = "indices"},
      %arg1: tensor<i64> {onnx.name = "depth"},
      %arg2: tensor<2xf32> {onnx.name = "values"})
      -> (tensor<2x10x2xf32> {onnx.name = "output"}) {
    %result = "onnx.OneHot"(%arg0, %arg1, %arg2) {axis = 1 : si64}
        : (tensor<2x2xi64>, tensor<i64>, tensor<2xf32>) -> tensor<2x10x2xf32>
    "onnx.Return"(%result) : (tensor<2x10x2xf32>) -> ()
  }

  // CHECK-LABEL: func.func @onehot_axis1
  // CHECK: hip.one_hot(%[[CTX:.*]]) ins(%[[IDX:.*]], %[[DEPTH:.*]], %[[VAL:.*]] : tensor<2x2xi64>, tensor<i64>, tensor<2xf32>) outs({{.*}} : tensor<2x10x2xf32>) {axis = 1 : i64}
  // CHECK-NOT: onnx.OneHot

  func.func @onehot_default_axis(
      %arg0: tensor<2xi64> {onnx.name = "indices"},
      %arg1: tensor<i64> {onnx.name = "depth"},
      %arg2: tensor<2xf32> {onnx.name = "values"})
      -> (tensor<2x4xf32> {onnx.name = "output"}) {
    %result = "onnx.OneHot"(%arg0, %arg1, %arg2)
        : (tensor<2xi64>, tensor<i64>, tensor<2xf32>) -> tensor<2x4xf32>
    "onnx.Return"(%result) : (tensor<2x4xf32>) -> ()
  }

  // CHECK-LABEL: func.func @onehot_default_axis
  // CHECK: hip.one_hot(%{{.*}}) ins({{.*}} : tensor<2xi64>, tensor<i64>, tensor<2xf32>) outs({{.*}} : tensor<2x4xf32>) {axis = -1 : i64}
}
