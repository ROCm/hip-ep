// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> {
    return %arg0 : tensor<1x128xf32>
  }

  func.func @reduce_mean_keepdims(%data: tensor<1x128xf32>, %axes: tensor<i64>) -> tensor<1x1xf32> {
    // CHECK-LABEL: func.func @reduce_mean_keepdims
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<1x128xf32>, %[[AXES:.*]]: tensor<i64>) -> tensor<1x1xf32>
    %output = "onnx.ReduceMean"(%data, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x128xf32>, tensor<i64>) -> tensor<1x1xf32>
    // CHECK-NOT: onnx.ReduceMean
    // CHECK: tensor.empty() : tensor<1x1xf32>
    // CHECK: hip.reduce_mean(%[[CTX]]) ins(%[[DATA]], %[[AXES]] : tensor<1x128xf32>, tensor<i64>) outs({{.*}} : tensor<1x1xf32>)
    return %output : tensor<1x1xf32>
  }

  func.func @reduce_mean_no_keepdims(%data: tensor<4x8xf32>, %axes: tensor<i64>) -> tensor<4xf32> {
    // CHECK-LABEL: func.func @reduce_mean_no_keepdims
    %output = "onnx.ReduceMean"(%data, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<4x8xf32>, tensor<i64>) -> tensor<4xf32>
    // CHECK: tensor.empty() : tensor<4xf32>
    // CHECK: hip.reduce_mean({{.*}}) ins({{.*}}) outs({{.*}} : tensor<4xf32>) {keepdims = 0 : i64}
    return %output : tensor<4xf32>
  }

  func.func @reduce_mean_dynamic(%data: tensor<?x?x512xf32>, %axes: tensor<i64>) -> tensor<?x?xf32> {
    // CHECK-LABEL: func.func @reduce_mean_dynamic
    // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[DATA3:.*]]: tensor<?x?x512xf32>, %[[AXES3:.*]]: tensor<i64>)
    %output = "onnx.ReduceMean"(%data, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x?x512xf32>, tensor<i64>) -> tensor<?x?xf32>
    // CHECK: tensor.dim
    // CHECK: tensor.empty({{.*}}) : tensor<?x?xf32>
    // CHECK: hip.reduce_mean(%[[CTX3]])
    return %output : tensor<?x?xf32>
  }
}
