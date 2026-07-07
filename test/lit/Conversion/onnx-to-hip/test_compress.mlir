// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @compress_axis0(
      %arg0: tensor<3x2xf32> {onnx.name = "input"},
      %arg1: tensor<3xi1> {onnx.name = "condition"})
      -> (tensor<2x2xf32> {onnx.name = "output"}) {
    %result = "onnx.Compress"(%arg0, %arg1) {axis = 0 : si64}
        : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<2x2xf32>
    "onnx.Return"(%result) : (tensor<2x2xf32>) -> ()
  }

  // CHECK-LABEL: func.func @compress_axis0
  // CHECK: hip.compress(%[[CTX:.*]]) ins(%[[IN:.*]], %[[COND:.*]] : tensor<3x2xf32>, tensor<3xi1>) outs({{.*}} : tensor<2x2xf32>) {axis = 0 : i64, flatten = false}
  // CHECK-NOT: onnx.Compress

  func.func @compress_flatten(
      %arg0: tensor<2x2xf32> {onnx.name = "input"},
      %arg1: tensor<3xi1> {onnx.name = "condition"})
      -> (tensor<2xf32> {onnx.name = "output"}) {
    %result = "onnx.Compress"(%arg0, %arg1)
        : (tensor<2x2xf32>, tensor<3xi1>) -> tensor<2xf32>
    "onnx.Return"(%result) : (tensor<2xf32>) -> ()
  }

  // CHECK-LABEL: func.func @compress_flatten
  // CHECK: hip.compress(%{{.*}}) ins({{.*}} : tensor<2x2xf32>, tensor<3xi1>) outs({{.*}} : tensor<2xf32>) {axis = 0 : i64, flatten = true}
}
