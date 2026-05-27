// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Shape is correctly lowered by ShapeToTensorDims to a mix of
// arith.constant (for static dims) and tensor.dim + arith.index_cast (for
// dynamic dims), wrapped in a tensor.from_elements.
//
// Coverage:
//   - all-static input  -> all arith.constant dims
//   - all-dynamic input -> all tensor.dim dims
//   - mixed shape       -> mix of constants + tensor.dim
//   - start/end attrs   -> sub-range of dims
//   - negative start/end -> normalized to (rank + value)
//   - empty range (start >= end) -> tensor<0xi64> constant
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16>) -> tensor<1x128x4096xf16> {
    return %arg0 : tensor<1x128x4096xf16>
  }

  // --- Fully static shape: all dims are arith.constant ---
  func.func @test_shape_static(%data: tensor<2x3x4xf16>) -> tensor<3xi64> {
    %result = "onnx.Shape"(%data) : (tensor<2x3x4xf16>) -> tensor<3xi64>
    return %result : tensor<3xi64>
  }

  // --- All dynamic dims: tensor.dim emitted for each ---
  func.func @test_shape_dynamic(%data: tensor<?x?x?xf16>) -> tensor<3xi64> {
    %result = "onnx.Shape"(%data) : (tensor<?x?x?xf16>) -> tensor<3xi64>
    return %result : tensor<3xi64>
  }

  // --- Mixed: dim 1 dynamic, dim 0/2 static ---
  func.func @test_shape_mixed(%data: tensor<2x?x4xf16>) -> tensor<3xi64> {
    %result = "onnx.Shape"(%data) : (tensor<2x?x4xf16>) -> tensor<3xi64>
    return %result : tensor<3xi64>
  }

  // --- start/end sub-range: only dims [1, 3) emitted ---
  func.func @test_shape_start_end(%data: tensor<2x3x4x5xf16>) -> tensor<2xi64> {
    %result = "onnx.Shape"(%data) {start = 1 : si64, end = 3 : si64}
        : (tensor<2x3x4x5xf16>) -> tensor<2xi64>
    return %result : tensor<2xi64>
  }

  // --- Negative end (-1 means rank-1 = 3): emits dims 0..2 ---
  func.func @test_shape_negative(%data: tensor<2x3x4x5xf16>) -> tensor<3xi64> {
    %result = "onnx.Shape"(%data) {start = 0 : si64, end = -1 : si64}
        : (tensor<2x3x4x5xf16>) -> tensor<3xi64>
    return %result : tensor<3xi64>
  }

  // --- Empty range (start >= end): 1-D tensor with zero elements ---
  func.func @test_shape_empty(%data: tensor<2x3x4xf16>) -> tensor<0xi64> {
    %result = "onnx.Shape"(%data) {start = 2 : si64, end = 1 : si64}
        : (tensor<2x3x4xf16>) -> tensor<0xi64>
    return %result : tensor<0xi64>
  }
}

// CHECK-LABEL: func.func @test_shape_static
// CHECK-NOT: onnx.Shape
// CHECK-NOT: tensor.dim
// CHECK-DAG: arith.constant 2 : i64
// CHECK-DAG: arith.constant 3 : i64
// CHECK-DAG: arith.constant 4 : i64
// CHECK: tensor.from_elements

// CHECK-LABEL: func.func @test_shape_dynamic
// CHECK-NOT: onnx.Shape
// CHECK: tensor.dim
// CHECK: arith.index_cast
// CHECK: tensor.dim
// CHECK: arith.index_cast
// CHECK: tensor.dim
// CHECK: arith.index_cast
// CHECK: tensor.from_elements

// CHECK-LABEL: func.func @test_shape_mixed
// CHECK-NOT: onnx.Shape
// CHECK-DAG: arith.constant 2 : i64
// CHECK-DAG: arith.constant 4 : i64
// CHECK-DAG: tensor.dim
// CHECK-DAG: arith.index_cast
// CHECK: tensor.from_elements

// CHECK-LABEL: func.func @test_shape_start_end
// CHECK-NOT: onnx.Shape
// CHECK-DAG: arith.constant 3 : i64
// CHECK-DAG: arith.constant 4 : i64
// CHECK-NOT: arith.constant 2 : i64
// CHECK-NOT: arith.constant 5 : i64
// CHECK: tensor.from_elements

// CHECK-LABEL: func.func @test_shape_negative
// CHECK-NOT: onnx.Shape
// CHECK-DAG: arith.constant 2 : i64
// CHECK-DAG: arith.constant 3 : i64
// CHECK-DAG: arith.constant 4 : i64
// CHECK-NOT: arith.constant 5 : i64
// CHECK: tensor.from_elements

// CHECK-LABEL: func.func @test_shape_empty
// CHECK-NOT: onnx.Shape
// CHECK-NOT: tensor.from_elements
// CHECK: arith.constant dense<> : tensor<0xi64>
