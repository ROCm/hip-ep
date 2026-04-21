// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: onnx.Range compile-time folding for i64 scalar constants.
//===----------------------------------------------------------------------===//
//
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1xf16>) -> tensor<1xf16> {
    return %arg0 : tensor<1xf16>
  }

  func.func @test_range_positive() -> tensor<4xi64> {
    %start = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %limit = "onnx.Constant"() {value = dense<8> : tensor<i64>} : () -> tensor<i64>
    %delta = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %0 = "onnx.Range"(%start, %limit, %delta) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
  }

  func.func @test_range_negative_delta() -> tensor<3xi64> {
    %start = "onnx.Constant"() {value = dense<5> : tensor<i64>} : () -> tensor<i64>
    %limit = "onnx.Constant"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
    %delta = "onnx.Constant"() {value = dense<-2> : tensor<i64>} : () -> tensor<i64>
    %0 = "onnx.Range"(%start, %limit, %delta) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<3xi64>
    return %0 : tensor<3xi64>
  }
}

// CHECK-LABEL: func.func @test_range_positive
// CHECK-NOT: onnx.Range
// CHECK: arith.constant 0 : i64
// CHECK: arith.constant 2 : i64
// CHECK: arith.constant 4 : i64
// CHECK: arith.constant 6 : i64
// CHECK: tensor.from_elements

// CHECK-LABEL: func.func @test_range_negative_delta
// CHECK-NOT: onnx.Range
// CHECK: arith.constant 5 : i64
// CHECK: arith.constant 3 : i64
// CHECK: arith.constant 1 : i64
// CHECK: tensor.from_elements
