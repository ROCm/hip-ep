// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify onnx.ReduceSumSquare is decomposed to its ONNX function body,
// hip.mul(x, x) followed by hip.reduce_sum, with the axes operand and all
// attributes passed through to the ReduceSum converter.
//
// Test cases:
// 1. rss_keepdims_axes_operand  — dynamic dims, constant axes operand, keepdims=1
// 2. rss_no_keepdims            — keepdims=0 drops the reduced axis
// 3. rss_axes_attribute         — opset-13 form, axes as an attribute
// 4. rss_runtime_axes           — axes is a runtime operand
// 5. rss_attributes_forwarded   — unknown and metadata attributes are forwarded
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128xf16>) -> tensor<1x128xf16> {
    return %arg0 : tensor<1x128xf16>
  }

  // --------------------------------------------------------------------------
  // 1. Per-head sum of squares over head_dim, as used by an L2 norm.
  // --------------------------------------------------------------------------
  func.func @rss_keepdims_axes_operand(%x: tensor<?x?x16x128xf16>) -> tensor<?x?x16x1xf16> {
    %axes = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %y = "onnx.ReduceSumSquare"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<?x?x16x128xf16>, tensor<1xi64>) -> tensor<?x?x16x1xf16>
    return %y : tensor<?x?x16x1xf16>
  }

  // CHECK-LABEL: func.func @rss_keepdims_axes_operand
  // CHECK-SAME: (%[[CTX1:.*]]: !hip.context, %[[X1:.*]]: tensor<?x?x16x128xf16>)
  // CHECK-NOT: onnx.ReduceSumSquare
  // CHECK: %[[SQ1:.*]] = hip.mul(%[[CTX1]]) ins(%[[X1]], %[[X1]] : tensor<?x?x16x128xf16>, tensor<?x?x16x128xf16>)
  // CHECK: tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x16x1xf16>
  // CHECK: hip.reduce_sum(%[[CTX1]]) ins(%[[SQ1]], %{{.*}} : tensor<?x?x16x128xf16>, tensor<1xi64>) outs(%{{.*}} : tensor<?x?x16x1xf16>) {reduced_axes = array<i64: 3>}

  // --------------------------------------------------------------------------
  // 2. keepdims = 0
  // --------------------------------------------------------------------------
  func.func @rss_no_keepdims(%x: tensor<4x8xf16>) -> tensor<4xf16> {
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %y = "onnx.ReduceSumSquare"(%x, %axes) {keepdims = 0 : si64}
      : (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4xf16>
    return %y : tensor<4xf16>
  }

  // CHECK-LABEL: func.func @rss_no_keepdims
  // CHECK: %[[SQ2:.*]] = hip.mul
  // CHECK: hip.reduce_sum({{.*}}) ins(%[[SQ2]], %{{.*}} : tensor<4x8xf16>, tensor<1xi64>) outs(%{{.*}} : tensor<4xf16>) {keepdims = 0 : i64, reduced_axes = array<i64: 1>}

  // --------------------------------------------------------------------------
  // 3. opset-13 form: axes is an attribute
  // --------------------------------------------------------------------------
  func.func @rss_axes_attribute(%x: tensor<2x3x5xf32>) -> tensor<2x1x5xf32> {
    %y = "onnx.ReduceSumSquare"(%x) {axes = [1], keepdims = 1 : si64}
      : (tensor<2x3x5xf32>) -> tensor<2x1x5xf32>
    return %y : tensor<2x1x5xf32>
  }

  // CHECK-LABEL: func.func @rss_axes_attribute
  // CHECK: %[[SQ3:.*]] = hip.mul
  // CHECK: hip.reduce_sum({{.*}}) ins(%[[SQ3]], %{{.*}} : tensor<2x3x5xf32>, tensor<1xi64>) outs(%{{.*}} : tensor<2x1x5xf32>) {reduced_axes = array<i64: 1>}

  // --------------------------------------------------------------------------
  // 4. axes known only at runtime: passed through, no reduced_axes
  // --------------------------------------------------------------------------
  func.func @rss_runtime_axes(%x: tensor<2x3x5xf16>, %axes: tensor<1xi64>) -> tensor<2x3x1xf16> {
    %y = "onnx.ReduceSumSquare"(%x, %axes) {keepdims = 1 : si64}
      : (tensor<2x3x5xf16>, tensor<1xi64>) -> tensor<2x3x1xf16>
    return %y : tensor<2x3x1xf16>
  }

  // CHECK-LABEL: func.func @rss_runtime_axes
  // CHECK-SAME: (%[[CTX4:.*]]: !hip.context, %[[X4:.*]]: tensor<2x3x5xf16>, %[[AX4:.*]]: tensor<1xi64>)
  // CHECK: %[[SQ4:.*]] = hip.mul
  // CHECK: hip.reduce_sum(%[[CTX4]]) ins(%[[SQ4]], %[[AX4]] : tensor<2x3x5xf16>, tensor<1xi64>) outs(%{{.*}} : tensor<2x3x1xf16>)
  // CHECK-NOT: reduced_axes
  // CHECK-NOT: onnx.ReduceSumSquare

  // --------------------------------------------------------------------------
  // 5. Every attribute, including ones the converter does not read, reaches
  //    onnx.ReduceSum. An unranked input keeps it unconverted so the
  //    forwarded attributes stay visible.
  // --------------------------------------------------------------------------
  func.func @rss_attributes_forwarded(%x: tensor<*xf16>) -> tensor<*xf16> {
    %y = "onnx.ReduceSumSquare"(%x) {future_attr = 1 : si64, keepdims = 0 : si64, onnx_node_name = "rss"}
      : (tensor<*xf16>) -> tensor<*xf16>
    return %y : tensor<*xf16>
  }

  // CHECK-LABEL: func.func @rss_attributes_forwarded
  // CHECK-NOT: onnx.ReduceSumSquare
  // CHECK: "onnx.ReduceSum"(%{{.*}}) {future_attr = 1 : si64, keepdims = 0 : si64, onnx_node_name = "rss"}
}
