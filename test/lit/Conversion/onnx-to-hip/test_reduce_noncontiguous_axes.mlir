// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Constant reduce axes: destination extents and non-contiguous axes.
//
// 1. mean_axes_2_4_keepdims0 — axes [2, 4] on ?x?x2x?x2xC with keepdims=0 is
//    a chain: reduce [4], then [2]. Each destination takes its dynamic
//    extents from the data dims that survive, not from the same position.
// 2. mean_keepdims0_dims      — one axis, keepdims=0: result dim 1 comes from
//    data dim 2.
// 3. max_axes_0_2_keepdims1   — chain with keepdims=1.
// 4. sum_axes_attribute       — opset-13 attribute axes [0, 2], chain.
// 5. l2_not_decomposed        — ReduceL2 is not separable: one op keeps
//    reduced_axes [1, 3] (rejected later by the lowering, not guessed).
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x4xf16>) -> tensor<1x4xf16> {
    return %arg0 : tensor<1x4xf16>
  }

  func.func @mean_axes_2_4_keepdims0(%x: tensor<?x?x2x?x2x16xf16>) -> tensor<?x?x?x16xf16> {
    %axes = "onnx.Constant"() {value = dense<[2, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
    %y = "onnx.ReduceMean"(%x, %axes) {keepdims = 0 : si64}
      : (tensor<?x?x2x?x2x16xf16>, tensor<2xi64>) -> tensor<?x?x?x16xf16>
    return %y : tensor<?x?x?x16xf16>
  }

  // CHECK-LABEL: func.func @mean_axes_2_4_keepdims0
  // CHECK-SAME: (%[[CTX1:.*]]: !hip.context, %[[X1:.*]]: tensor<?x?x2x?x2x16xf16>)
  // CHECK-DAG: %[[A0:.*]] = tensor.dim %[[X1]], %c0
  // CHECK-DAG: %[[A1:.*]] = tensor.dim %[[X1]], %c1
  // CHECK-DAG: %[[A3:.*]] = tensor.dim %[[X1]], %c3
  // CHECK: %[[I1:.*]] = tensor.empty(%[[A0]], %[[A1]], %[[A3]]) : tensor<?x?x2x?x16xf16>
  // CHECK: %[[T1:.*]] = hip.reduce_mean(%[[CTX1]]) ins(%[[X1]], %{{.*}} : tensor<?x?x2x?x2x16xf16>, tensor<1xi64>) outs(%[[I1]] : tensor<?x?x2x?x16xf16>) {keepdims = 0 : i64, reduced_axes = array<i64: 4>}
  // CHECK-DAG: %[[B0:.*]] = tensor.dim %[[T1]], %c0
  // CHECK-DAG: %[[B1:.*]] = tensor.dim %[[T1]], %c1
  // CHECK-DAG: %[[B3:.*]] = tensor.dim %[[T1]], %c3
  // CHECK: %[[I2:.*]] = tensor.empty(%[[B0]], %[[B1]], %[[B3]]) : tensor<?x?x?x16xf16>
  // CHECK: %[[Y1:.*]] = hip.reduce_mean(%[[CTX1]]) ins(%[[T1]], %{{.*}} : tensor<?x?x2x?x16xf16>, tensor<1xi64>) outs(%[[I2]] : tensor<?x?x?x16xf16>) {keepdims = 0 : i64, reduced_axes = array<i64: 2>}
  // CHECK: return %[[Y1]]

  func.func @mean_keepdims0_dims(%x: tensor<?x2x?x8xf16>) -> tensor<?x?x8xf16> {
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %y = "onnx.ReduceMean"(%x, %axes) {keepdims = 0 : si64}
      : (tensor<?x2x?x8xf16>, tensor<1xi64>) -> tensor<?x?x8xf16>
    return %y : tensor<?x?x8xf16>
  }

  // CHECK-LABEL: func.func @mean_keepdims0_dims
  // CHECK-SAME: (%{{.*}}: !hip.context, %[[X2:.*]]: tensor<?x2x?x8xf16>)
  // CHECK-DAG: %[[D0:.*]] = tensor.dim %[[X2]], %c0
  // CHECK-DAG: %[[D2:.*]] = tensor.dim %[[X2]], %c2
  // CHECK: tensor.empty(%[[D0]], %[[D2]]) : tensor<?x?x8xf16>
  // CHECK: hip.reduce_mean({{.*}}) {keepdims = 0 : i64, reduced_axes = array<i64: 1>}

  func.func @max_axes_0_2_keepdims1(%x: tensor<4x3x5xf16>) -> tensor<1x3x1xf16> {
    %axes = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64>} : () -> tensor<2xi64>
    %y = "onnx.ReduceMax"(%x, %axes) {keepdims = 1 : si64}
      : (tensor<4x3x5xf16>, tensor<2xi64>) -> tensor<1x3x1xf16>
    return %y : tensor<1x3x1xf16>
  }

  // CHECK-LABEL: func.func @max_axes_0_2_keepdims1
  // CHECK: %[[M1:.*]] = hip.reduce_max({{.*}}) outs(%{{.*}} : tensor<4x3x1xf16>) {reduced_axes = array<i64: 2>}
  // CHECK: hip.reduce_max({{.*}}) ins(%[[M1]], %{{.*}} : tensor<4x3x1xf16>, tensor<1xi64>) outs(%{{.*}} : tensor<1x3x1xf16>) {reduced_axes = array<i64: 0>}

  func.func @sum_axes_attribute(%x: tensor<4x3x5xf32>) -> tensor<3xf32> {
    %y = "onnx.ReduceSum"(%x) {axes = [0, 2], keepdims = 0 : si64}
      : (tensor<4x3x5xf32>) -> tensor<3xf32>
    return %y : tensor<3xf32>
  }

  // CHECK-LABEL: func.func @sum_axes_attribute
  // CHECK: %[[S1:.*]] = hip.reduce_sum({{.*}}) outs(%{{.*}} : tensor<4x3xf32>) {keepdims = 0 : i64, reduced_axes = array<i64: 2>}
  // CHECK: hip.reduce_sum({{.*}}) ins(%[[S1]], %{{.*}} : tensor<4x3xf32>, tensor<1xi64>) outs(%{{.*}} : tensor<3xf32>) {keepdims = 0 : i64, reduced_axes = array<i64: 0>}

  func.func @l2_not_decomposed(%x: tensor<2x3x4x5xf16>) -> tensor<2x4xf16> {
    %axes = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
    %y = "onnx.ReduceL2"(%x, %axes) {keepdims = 0 : si64}
      : (tensor<2x3x4x5xf16>, tensor<2xi64>) -> tensor<2x4xf16>
    return %y : tensor<2x4xf16>
  }

  // CHECK-LABEL: func.func @l2_not_decomposed
  // CHECK: hip.reduce_l2({{.*}}) outs(%{{.*}} : tensor<2x4xf16>) {keepdims = 0 : i64, reduced_axes = array<i64: 1, 3>}
  // CHECK-NOT: hip.reduce_l2
}
