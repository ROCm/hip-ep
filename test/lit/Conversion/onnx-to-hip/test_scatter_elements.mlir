// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    return %arg0 : tensor<2x2xf32>
  }

  func.func @scatter_elements_axis1(
      %arg0: tensor<2x2xf32>,
      %arg1: tensor<2x2xi32>,
      %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %result = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 1 : si64}
        : (tensor<2x2xf32>, tensor<2x2xi32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %result : tensor<2x2xf32>
  }

  // CHECK-LABEL: func.func @scatter_elements_axis1
  // CHECK: hip.scatter_elements(%[[CTX:.*]]) ins(%[[DATA:.*]], %[[IDX:.*]], %[[UPD:.*]] : tensor<2x2xf32>, tensor<2x2xi32>, tensor<2x2xf32>) outs({{.*}} : tensor<2x2xf32>) {axis = 1 : i64} : tensor<2x2xf32>
  // CHECK-NOT: onnx.ScatterElements

  func.func @scatter_elements_add(
      %arg0: tensor<3x2xf32>,
      %arg1: tensor<3x2xi64>,
      %arg2: tensor<3x2xf32>) -> tensor<3x2xf32> {
    %result = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {
      axis = 0 : si64,
      reduction = "add"
    } : (tensor<3x2xf32>, tensor<3x2xi64>, tensor<3x2xf32>) -> tensor<3x2xf32>
    return %result : tensor<3x2xf32>
  }

  // CHECK-LABEL: func.func @scatter_elements_add
  // CHECK: hip.scatter_elements(%{{.*}}) ins({{.*}} : tensor<3x2xf32>, tensor<3x2xi64>, tensor<3x2xf32>) outs({{.*}} : tensor<3x2xf32>) {reduction = "add"} : tensor<3x2xf32>

  func.func @scatter_elements_dynamic(
      %data: tensor<?x4xf32>,
      %indices: tensor<2x3xi64>,
      %updates: tensor<2x3xf32>) -> tensor<?x4xf32> {
    %result = "onnx.ScatterElements"(%data, %indices, %updates)
        {axis = 1 : si64}
        : (tensor<?x4xf32>, tensor<2x3xi64>, tensor<2x3xf32>)
          -> tensor<?x4xf32>
    return %result : tensor<?x4xf32>
  }

  // CHECK-LABEL: func.func @scatter_elements_dynamic
  // CHECK-SAME: (%{{[^,]*}}, %[[DATA:[A-Za-z0-9_]+]]: tensor<?x4xf32>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[D0:.*]] = tensor.dim %[[DATA]], %[[C0]]
  // CHECK: %[[EMPTY:.*]] = tensor.empty(%[[D0]]) : tensor<?x4xf32>
  // CHECK: hip.scatter_elements(%{{[^)]*}})
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<?x4xf32>)
}
