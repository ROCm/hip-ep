// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Unsqueeze is correctly lowered to tensor.expand_shape
// (zero-cost standard MLIR metadata op).
//
// Unsqueeze inserts dimensions of size 1 at specified axes. Like Reshape,
// it's a zero-cost operation that reinterprets shape/strides without
// moving data. No custom HIP dialect op or kernel is needed.
//
// This test validates:
// - Insert single dim at beginning:  2D -> 3D  [128, 4096] -> [1, 128, 4096]
// - Insert single dim at end:        2D -> 3D  [128, 4096] -> [128, 4096, 1]
// - Insert single dim in middle:     2D -> 3D  [128, 4096] -> [128, 1, 4096]
// - Insert multiple dims:            2D -> 4D  [128, 4096] -> [1, 128, 1, 4096]
// - Negative axis handling:          2D -> 3D  [128, 4096] -> [128, 4096, 1] (axis=-1)
//
// Each test checks:
// - onnx.Unsqueeze is removed
// - tensor.expand_shape is generated with correct reassociation indices
// - Output shape matches expected dimensions
//
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<128x4096xf16>) -> tensor<128x4096xf16> {
    return %arg0 : tensor<128x4096xf16>
  }

  // --- Insert dimension at beginning ---
  func.func @test_unsqueeze_front(%data: tensor<128x4096xf16>) -> tensor<1x128x4096xf16> {
    %axes = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.Unsqueeze"(%data, %axes) : (tensor<128x4096xf16>, tensor<1xi64>) -> tensor<1x128x4096xf16>
    return %result : tensor<1x128x4096xf16>
  }

  // --- Insert dimension at end ---
  func.func @test_unsqueeze_back(%data: tensor<128x4096xf16>) -> tensor<128x4096x1xf16> {
    %axes = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.Unsqueeze"(%data, %axes) : (tensor<128x4096xf16>, tensor<1xi64>) -> tensor<128x4096x1xf16>
    return %result : tensor<128x4096x1xf16>
  }

  // --- Insert dimension in middle ---
  func.func @test_unsqueeze_middle(%data: tensor<128x4096xf16>) -> tensor<128x1x4096xf16> {
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.Unsqueeze"(%data, %axes) : (tensor<128x4096xf16>, tensor<1xi64>) -> tensor<128x1x4096xf16>
    return %result : tensor<128x1x4096xf16>
  }

  // --- Insert multiple dimensions ---
  func.func @test_unsqueeze_multiple(%data: tensor<128x4096xf16>) -> tensor<1x128x1x4096xf16> {
    %axes = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %result = "onnx.Unsqueeze"(%data, %axes) : (tensor<128x4096xf16>, tensor<2xi64>) -> tensor<1x128x1x4096xf16>
    return %result : tensor<1x128x1x4096xf16>
  }

  // --- Insert dimension with negative axis ---
  func.func @test_unsqueeze_negative_axis(%data: tensor<128x4096xf16>) -> tensor<128x4096x1xf16> {
    %axes = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.Unsqueeze"(%data, %axes) : (tensor<128x4096xf16>, tensor<1xi64>) -> tensor<128x4096x1xf16>
    return %result : tensor<128x4096x1xf16>
  }

  // --- Block-argument axes (graph-slice boundary) + dynamic leading dims ---
  // The axes VALUE is never read; the reassociation is derived from shapes, so
  // a non-constant (block-argument) axes operand -- as produced when a graph
  // slice routes a folded axes constant across a part boundary -- must still
  // lower to a zero-cost expand_shape rather than surviving unconverted.
  func.func @test_unsqueeze_blockarg_axes(%data: tensor<?x?x128xf16>, %axes: tensor<1xi64>) -> tensor<?x?x128x1xf16> {
    %result = "onnx.Unsqueeze"(%data, %axes) : (tensor<?x?x128xf16>, tensor<1xi64>) -> tensor<?x?x128x1xf16>
    return %result : tensor<?x?x128x1xf16>
  }
}

// CHECK-LABEL: func.func @test_unsqueeze_front
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<128x4096xf16>)
// CHECK-NOT: onnx.Unsqueeze
// CHECK: %[[RESULT:.*]] = tensor.expand_shape %[[DATA]] {{\[\[}}0, 1], [2]] output_shape [1, 128, 4096] : tensor<128x4096xf16> into tensor<1x128x4096xf16>
// CHECK: return %[[RESULT]] : tensor<1x128x4096xf16>

// CHECK-LABEL: func.func @test_unsqueeze_back
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<128x4096xf16>)
// CHECK-NOT: onnx.Unsqueeze
// CHECK: %[[RESULT:.*]] = tensor.expand_shape %[[DATA]] {{\[\[}}0], [1, 2]] output_shape [128, 4096, 1] : tensor<128x4096xf16> into tensor<128x4096x1xf16>
// CHECK: return %[[RESULT]] : tensor<128x4096x1xf16>

// CHECK-LABEL: func.func @test_unsqueeze_middle
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<128x4096xf16>)
// CHECK-NOT: onnx.Unsqueeze
// CHECK: %[[RESULT:.*]] = tensor.expand_shape %[[DATA]] {{\[\[}}0], [1, 2]] output_shape [128, 1, 4096] : tensor<128x4096xf16> into tensor<128x1x4096xf16>
// CHECK: return %[[RESULT]] : tensor<128x1x4096xf16>

// CHECK-LABEL: func.func @test_unsqueeze_multiple
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<128x4096xf16>)
// CHECK-NOT: onnx.Unsqueeze
// CHECK: %[[RESULT:.*]] = tensor.expand_shape %[[DATA]] {{\[\[}}0, 1], [2, 3]] output_shape [1, 128, 1, 4096] : tensor<128x4096xf16> into tensor<1x128x1x4096xf16>
// CHECK: return %[[RESULT]] : tensor<1x128x1x4096xf16>

// CHECK-LABEL: func.func @test_unsqueeze_negative_axis
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<128x4096xf16>)
// CHECK-NOT: onnx.Unsqueeze
// CHECK: %[[RESULT:.*]] = tensor.expand_shape %[[DATA]] {{\[\[}}0], [1, 2]] output_shape [128, 4096, 1] : tensor<128x4096xf16> into tensor<128x4096x1xf16>
// CHECK: return %[[RESULT]] : tensor<128x4096x1xf16>

// CHECK-LABEL: func.func @test_unsqueeze_blockarg_axes
// CHECK-NOT: onnx.Unsqueeze
// CHECK: %[[RESULT:.*]] = tensor.expand_shape %{{.*}} {{\[\[}}0], [1], [2, 3]] output_shape [%{{.*}}, %{{.*}}, 128, 1] : tensor<?x?x128xf16> into tensor<?x?x128x1xf16>
// CHECK: return %[[RESULT]] : tensor<?x?x128x1xf16>
