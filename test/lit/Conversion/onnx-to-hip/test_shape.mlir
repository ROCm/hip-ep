// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: onnx.Shape compile-time handling.
//
// Verifies:
// - Static-ranked input is folded to arith.constant.
// - Dynamic input dimension is not folded and onnx.Shape remains.
// - start attribute is honored for slicing shape output.
//===----------------------------------------------------------------------===//
//
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x3x4xf16>) -> tensor<2x3x4xf16> {
    return %arg0 : tensor<2x3x4xf16>
  }

  func.func @test_shape_static(%data: tensor<2x3x4xf16>) -> tensor<3xi64> {
    %0 = "onnx.Shape"(%data) : (tensor<2x3x4xf16>) -> tensor<3xi64>
    return %0 : tensor<3xi64>
  }

  func.func @test_shape_with_start(%data: tensor<2x3x4xf16>) -> tensor<2xi64> {
    %0 = "onnx.Shape"(%data) {start = 1 : i64} : (tensor<2x3x4xf16>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }

  func.func @test_shape_dynamic(%data: tensor<?x3x4xf16>) -> tensor<3xi64> {
    %0 = "onnx.Shape"(%data) : (tensor<?x3x4xf16>) -> tensor<3xi64>
    return %0 : tensor<3xi64>
  }
}

// CHECK-LABEL: func.func @test_shape_static
// CHECK-NOT: onnx.Shape
// CHECK: arith.constant 2 : i64
// CHECK: arith.constant 3 : i64
// CHECK: arith.constant 4 : i64
// CHECK: tensor.from_elements

// CHECK-LABEL: func.func @test_shape_with_start
// CHECK-NOT: onnx.Shape
// CHECK: arith.constant 3 : i64
// CHECK: arith.constant 4 : i64
// CHECK: tensor.from_elements

// CHECK-LABEL: func.func @test_shape_dynamic
// CHECK: "onnx.Shape"
