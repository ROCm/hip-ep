// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> {
    return %arg0 : tensor<3x2xf32>
  }

  func.func @compress_axis0(
      %arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<2x2xf32> {
    %result = "onnx.Compress"(%arg0, %arg1) {axis = 0 : si64}
        : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<2x2xf32>
    return %result : tensor<2x2xf32>
  }

  // CHECK-LABEL: func.func @compress_axis0
  // CHECK: hip.compress(%[[CTX:.*]]) ins(%[[IN:.*]], %[[COND:.*]] : tensor<3x2xf32>, tensor<3xi1>) outs({{.*}} : tensor<2x2xf32>) : tensor<2x2xf32>
  // CHECK-NOT: onnx.Compress

  func.func @compress_flatten(
      %arg0: tensor<2x2xf32>, %arg1: tensor<3xi1>) -> tensor<2xf32> {
    %result = "onnx.Compress"(%arg0, %arg1)
        : (tensor<2x2xf32>, tensor<3xi1>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @compress_flatten
  // CHECK: hip.compress(%{{.*}}) ins({{.*}} : tensor<2x2xf32>, tensor<3xi1>) outs({{.*}} : tensor<2xf32>) {flatten = true} : tensor<2xf32>
}
