// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The tensor.empty init of ReduceSum / ReduceMean / ReduceMax / ReduceMin
// takes each dynamic result size from the input dim it maps to:
// - keepdims = 0 over constant axis 0 of tensor<2x?xf32>: the surviving
//   dynamic result dim 0 is input dim 1, not input dim 0.
// - keepdims = 1 with the reduced dim declared dynamic: its extent is 1.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @sum_drop(%x: tensor<2x?xf32>) -> tensor<?xf32> {
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %r = "onnx.ReduceSum"(%x, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x?xf32>, tensor<1xi64>) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @sum_drop
  // CHECK-SAME: %[[X:[a-z0-9_]+]]: tensor<2x?xf32>
  // CHECK: %[[D:.*]] = tensor.dim %[[X]], %c1 : tensor<2x?xf32>
  // CHECK: tensor.empty(%[[D]]) : tensor<?xf32>
  // CHECK: hip.reduce_sum

  func.func @sum_keep_dynamic(%x: tensor<?x4xf32>) -> tensor<?x4xf32> {
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %r = "onnx.ReduceSum"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x4xf32>, tensor<1xi64>) -> tensor<?x4xf32>
    return %r : tensor<?x4xf32>
  }
  // CHECK-LABEL: func.func @sum_keep_dynamic
  // CHECK-NOT: tensor.dim
  // CHECK: tensor.empty(%c1) : tensor<?x4xf32>
  // CHECK: hip.reduce_sum

  func.func @mean_drop(%x: tensor<2x?xf32>) -> tensor<?xf32> {
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %r = "onnx.ReduceMean"(%x, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x?xf32>, tensor<1xi64>) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @mean_drop
  // CHECK-SAME: %[[X:[a-z0-9_]+]]: tensor<2x?xf32>
  // CHECK: %[[D:.*]] = tensor.dim %[[X]], %c1 : tensor<2x?xf32>
  // CHECK: tensor.empty(%[[D]]) : tensor<?xf32>
  // CHECK: hip.reduce_mean

  func.func @mean_keep_dynamic(%x: tensor<?x4xf32>) -> tensor<?x4xf32> {
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %r = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x4xf32>, tensor<1xi64>) -> tensor<?x4xf32>
    return %r : tensor<?x4xf32>
  }
  // CHECK-LABEL: func.func @mean_keep_dynamic
  // CHECK-NOT: tensor.dim
  // CHECK: tensor.empty(%c1) : tensor<?x4xf32>
  // CHECK: hip.reduce_mean

  func.func @max_drop(%x: tensor<2x?xf32>) -> tensor<?xf32> {
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %r = "onnx.ReduceMax"(%x, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x?xf32>, tensor<1xi64>) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @max_drop
  // CHECK-SAME: %[[X:[a-z0-9_]+]]: tensor<2x?xf32>
  // CHECK: %[[D:.*]] = tensor.dim %[[X]], %c1 : tensor<2x?xf32>
  // CHECK: tensor.empty(%[[D]]) : tensor<?xf32>
  // CHECK: hip.reduce_max

  func.func @max_keep_dynamic(%x: tensor<?x4xf32>) -> tensor<?x4xf32> {
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %r = "onnx.ReduceMax"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x4xf32>, tensor<1xi64>) -> tensor<?x4xf32>
    return %r : tensor<?x4xf32>
  }
  // CHECK-LABEL: func.func @max_keep_dynamic
  // CHECK-NOT: tensor.dim
  // CHECK: tensor.empty(%c1) : tensor<?x4xf32>
  // CHECK: hip.reduce_max

  func.func @min_drop(%x: tensor<2x?xf32>) -> tensor<?xf32> {
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %r = "onnx.ReduceMin"(%x, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x?xf32>, tensor<1xi64>) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @min_drop
  // CHECK-SAME: %[[X:[a-z0-9_]+]]: tensor<2x?xf32>
  // CHECK: %[[D:.*]] = tensor.dim %[[X]], %c1 : tensor<2x?xf32>
  // CHECK: tensor.empty(%[[D]]) : tensor<?xf32>
  // CHECK: hip.reduce_min

  func.func @min_keep_dynamic(%x: tensor<?x4xf32>) -> tensor<?x4xf32> {
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %r = "onnx.ReduceMin"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x4xf32>, tensor<1xi64>) -> tensor<?x4xf32>
    return %r : tensor<?x4xf32>
  }
  // CHECK-LABEL: func.func @min_keep_dynamic
  // CHECK-NOT: tensor.dim
  // CHECK: tensor.empty(%c1) : tensor<?x4xf32>
  // CHECK: hip.reduce_min
}
