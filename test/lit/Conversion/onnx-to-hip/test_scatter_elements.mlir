// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @scatter_elements_axis1(
      %arg0: tensor<2x2xf32> {onnx.name = "data"},
      %arg1: tensor<2x2xi32> {onnx.name = "indices"},
      %arg2: tensor<2x2xf32> {onnx.name = "updates"})
      -> (tensor<2x2xf32> {onnx.name = "output"}) {
    %result = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 1 : si64}
        : (tensor<2x2xf32>, tensor<2x2xi32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "onnx.Return"(%result) : (tensor<2x2xf32>) -> ()
  }

  // CHECK-LABEL: func.func @scatter_elements_axis1
  // CHECK: hip.scatter_elements(%[[CTX:.*]]) ins(%[[DATA:.*]], %[[IDX:.*]], %[[UPD:.*]] : tensor<2x2xf32>, tensor<2x2xi32>, tensor<2x2xf32>) outs({{.*}} : tensor<2x2xf32>) {axis = 1 : i64, reduction = "none"}
  // CHECK-NOT: onnx.ScatterElements

  func.func @scatter_elements_add(
      %arg0: tensor<3x2xf32> {onnx.name = "data"},
      %arg1: tensor<3x2xi64> {onnx.name = "indices"},
      %arg2: tensor<3x2xf32> {onnx.name = "updates"})
      -> (tensor<3x2xf32> {onnx.name = "output"}) {
    %result = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {
      axis = 0 : si64,
      reduction = "add"
    } : (tensor<3x2xf32>, tensor<3x2xi64>, tensor<3x2xf32>) -> tensor<3x2xf32>
    "onnx.Return"(%result) : (tensor<3x2xf32>) -> ()
  }

  // CHECK-LABEL: func.func @scatter_elements_add
  // CHECK: hip.scatter_elements(%{{.*}}) ins({{.*}} : tensor<3x2xf32>, tensor<3x2xi64>, tensor<3x2xf32>) outs({{.*}} : tensor<3x2xf32>) {axis = 0 : i64, reduction = "add"}
}
