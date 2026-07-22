// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    return %arg0 : tensor<2x2xf32>
  }

  func.func @gather_elements_axis1(
      %data: tensor<2x2xf32>, %indices: tensor<2x2xi32>) -> tensor<2x2xf32> {
    %result = "onnx.GatherElements"(%data, %indices) {axis = 1 : si64}
        : (tensor<2x2xf32>, tensor<2x2xi32>) -> tensor<2x2xf32>
    return %result : tensor<2x2xf32>
  }

  // CHECK-LABEL: func.func @gather_elements_axis1
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<2x2xf32>, %[[IDX:.*]]: tensor<2x2xi32>)
  // CHECK: tensor.empty() : tensor<2x2xf32>
  // CHECK: hip.gather_elements(%[[CTX]]) ins(%[[DATA]], %[[IDX]] : tensor<2x2xf32>, tensor<2x2xi32>) outs({{.*}} : tensor<2x2xf32>) {axis = 1 : i64}
  // CHECK-NOT: onnx.GatherElements

  func.func @gather_elements_axis0(
      %data: tensor<3x2xf32>, %indices: tensor<3x2xi64>) -> tensor<3x2xf32> {
    %result = "onnx.GatherElements"(%data, %indices) {axis = 0 : si64}
        : (tensor<3x2xf32>, tensor<3x2xi64>) -> tensor<3x2xf32>
    return %result : tensor<3x2xf32>
  }

  // CHECK-LABEL: func.func @gather_elements_axis0
  // CHECK: hip.gather_elements(%{{.*}}) ins({{.*}} : tensor<3x2xf32>, tensor<3x2xi64>) outs({{.*}} : tensor<3x2xf32>) : tensor<3x2xf32>

  func.func @gather_elements_dynamic(
      %data: tensor<?x?xf16>, %indices: tensor<?x?xi32>) -> tensor<?x?xf16> {
    %result = "onnx.GatherElements"(%data, %indices) {axis = -1 : si64}
        : (tensor<?x?xf16>, tensor<?x?xi32>) -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }

  // CHECK-LABEL: func.func @gather_elements_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[DATA3:.*]]: tensor<?x?xf16>, %[[IDX3:.*]]: tensor<?x?xi32>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[IDX3]], %[[C0]]
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[IDX3]], %[[C1]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf16>
  // CHECK: hip.gather_elements(%[[CTX3]]) ins(%[[DATA3]], %[[IDX3]] : tensor<?x?xf16>, tensor<?x?xi32>) outs(%[[INIT]] : tensor<?x?xf16>) {axis = -1 : i64} : tensor<?x?xf16>
}
