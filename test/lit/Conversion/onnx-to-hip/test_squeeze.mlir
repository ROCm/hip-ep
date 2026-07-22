// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Squeeze is correctly lowered to tensor.collapse_shape
// (zero-cost standard MLIR metadata op).
//
// Squeeze removes dimensions of size 1 at specified axes. Like Reshape and
// Unsqueeze, it's a zero-cost operation that reinterprets shape/strides
// without moving data. No custom HIP dialect op or kernel is needed.
//
// This test validates:
// - Remove single dim from front:  3D -> 2D  [1, 128, 4096] -> [128, 4096]
// - Remove single dim from end:    3D -> 2D  [128, 4096, 1] -> [128, 4096]
// - Remove single dim from middle: 3D -> 2D  [128, 1, 4096] -> [128, 4096]
// - Remove multiple dims:          4D -> 2D  [1, 128, 1, 4096] -> [128, 4096]
// - Negative axis handling:        3D -> 2D  [128, 4096, 1] -> [128, 4096] (axis=-1)
//
// Each test checks:
// - onnx.Squeeze is removed
// - tensor.collapse_shape is generated with correct reassociation indices
// - Output shape matches expected dimensions
//
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<128x4096xf16>) -> tensor<128x4096xf16> {
    return %arg0 : tensor<128x4096xf16>
  }

  // --- Remove dimension from beginning ---
  func.func @test_squeeze_front(%data: tensor<1x128x4096xf16>) -> tensor<128x4096xf16> {
    %axes = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.Squeeze"(%data, %axes) : (tensor<1x128x4096xf16>, tensor<1xi64>) -> tensor<128x4096xf16>
    return %result : tensor<128x4096xf16>
  }

  // --- Remove dimension from end ---
  func.func @test_squeeze_back(%data: tensor<128x4096x1xf16>) -> tensor<128x4096xf16> {
    %axes = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.Squeeze"(%data, %axes) : (tensor<128x4096x1xf16>, tensor<1xi64>) -> tensor<128x4096xf16>
    return %result : tensor<128x4096xf16>
  }

  // --- Remove dimension from middle ---
  func.func @test_squeeze_middle(%data: tensor<128x1x4096xf16>) -> tensor<128x4096xf16> {
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.Squeeze"(%data, %axes) : (tensor<128x1x4096xf16>, tensor<1xi64>) -> tensor<128x4096xf16>
    return %result : tensor<128x4096xf16>
  }

  // --- Remove multiple dimensions ---
  func.func @test_squeeze_multiple(%data: tensor<1x128x1x4096xf16>) -> tensor<128x4096xf16> {
    %axes = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %result = "onnx.Squeeze"(%data, %axes) : (tensor<1x128x1x4096xf16>, tensor<2xi64>) -> tensor<128x4096xf16>
    return %result : tensor<128x4096xf16>
  }

  // --- Remove dimension with negative axis ---
  func.func @test_squeeze_negative_axis(%data: tensor<128x4096x1xf16>) -> tensor<128x4096xf16> {
    %axes = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.Squeeze"(%data, %axes) : (tensor<128x4096x1xf16>, tensor<1xi64>) -> tensor<128x4096xf16>
    return %result : tensor<128x4096xf16>
  }

  // --- Squeeze [B,1] -> rank-0 scalar, dynamic batch (pooler statistic) ---
  // collapse_shape into rank-0 needs static all-ones input, so the dynamic
  // batch dim is sliced to a unit corner first.
  func.func @test_squeeze_to_scalar_dyn(%data: tensor<?x1xf16>) -> tensor<f16> {
    %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
    %result = "onnx.Squeeze"(%data, %axes) : (tensor<?x1xf16>, tensor<2xi64>) -> tensor<f16>
    return %result : tensor<f16>
  }

  // --- Squeeze [1,1] -> rank-0 scalar, fully static ---
  func.func @test_squeeze_to_scalar_static(%data: tensor<1x1xf16>) -> tensor<f16> {
    %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
    %result = "onnx.Squeeze"(%data, %axes) : (tensor<1x1xf16>, tensor<2xi64>) -> tensor<f16>
    return %result : tensor<f16>
  }
}

// CHECK-LABEL: func.func @test_squeeze_front
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<1x128x4096xf16>)
// CHECK-NOT: onnx.Squeeze
// CHECK: %[[RESULT:.*]] = tensor.collapse_shape %[[DATA]] {{\[\[}}0, 1], [2]] : tensor<1x128x4096xf16> into tensor<128x4096xf16>
// CHECK: return %[[RESULT]] : tensor<128x4096xf16>

// CHECK-LABEL: func.func @test_squeeze_back
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<128x4096x1xf16>)
// CHECK-NOT: onnx.Squeeze
// CHECK: %[[RESULT:.*]] = tensor.collapse_shape %[[DATA]] {{\[\[}}0], [1, 2]] : tensor<128x4096x1xf16> into tensor<128x4096xf16>
// CHECK: return %[[RESULT]] : tensor<128x4096xf16>

// CHECK-LABEL: func.func @test_squeeze_middle
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<128x1x4096xf16>)
// CHECK-NOT: onnx.Squeeze
// CHECK: %[[RESULT:.*]] = tensor.collapse_shape %[[DATA]] {{\[\[}}0], [1, 2]] : tensor<128x1x4096xf16> into tensor<128x4096xf16>
// CHECK: return %[[RESULT]] : tensor<128x4096xf16>

// CHECK-LABEL: func.func @test_squeeze_multiple
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<1x128x1x4096xf16>)
// CHECK-NOT: onnx.Squeeze
// CHECK: %[[RESULT:.*]] = tensor.collapse_shape %[[DATA]] {{\[\[}}0, 1], [2, 3]] : tensor<1x128x1x4096xf16> into tensor<128x4096xf16>
// CHECK: return %[[RESULT]] : tensor<128x4096xf16>

// CHECK-LABEL: func.func @test_squeeze_negative_axis
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<128x4096x1xf16>)
// CHECK-NOT: onnx.Squeeze
// CHECK: %[[RESULT:.*]] = tensor.collapse_shape %[[DATA]] {{\[\[}}0], [1, 2]] : tensor<128x4096x1xf16> into tensor<128x4096xf16>
// CHECK: return %[[RESULT]] : tensor<128x4096xf16>

// CHECK-LABEL: func.func @test_squeeze_to_scalar_dyn
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<?x1xf16>)
// CHECK-NOT: onnx.Squeeze
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[DATA]][0, 0] [1, 1] [1, 1] : tensor<?x1xf16> to tensor<1x1xf16>
// CHECK: %[[RESULT:.*]] = tensor.collapse_shape %[[SLICE]] [] : tensor<1x1xf16> into tensor<f16>
// CHECK: return %[[RESULT]] : tensor<f16>

// CHECK-LABEL: func.func @test_squeeze_to_scalar_static
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<1x1xf16>)
// CHECK-NOT: onnx.Squeeze
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[DATA]][0, 0] [1, 1] [1, 1] : tensor<1x1xf16> to tensor<1x1xf16>
// CHECK: %[[RESULT:.*]] = tensor.collapse_shape %[[SLICE]] [] : tensor<1x1xf16> into tensor<f16>
// CHECK: return %[[RESULT]] : tensor<f16>
