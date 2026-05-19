// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify that onnx.ConstantOfShape is folded to a splat arith.constant
// at compile time when its shape input is a compile-time constant.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: default value (fp32 zero) with shape from arith.constant.
  func.func @test_constant_of_shape_default() -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @test_constant_of_shape_default
    %shape = arith.constant dense<[2, 3]> : tensor<2xi64>
    %r = "onnx.ConstantOfShape"(%shape) : (tensor<2xi64>) -> tensor<2x3xf32>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK: arith.constant dense<0.000000e+00> : tensor<2x3xf32>

    return %r : tensor<2x3xf32>
  }

  // Test 2: custom fp32 value attribute.
  func.func @test_constant_of_shape_value_f32() -> tensor<2x2xf32> {
    // CHECK-LABEL: func.func @test_constant_of_shape_value_f32
    %shape = arith.constant dense<[2, 2]> : tensor<2xi64>
    %r = "onnx.ConstantOfShape"(%shape) {
      value = dense<1.500000e+00> : tensor<1xf32>
    } : (tensor<2xi64>) -> tensor<2x2xf32>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK: arith.constant dense<1.500000e+00> : tensor<2x2xf32>

    return %r : tensor<2x2xf32>
  }

  // Test 3: int64 value attribute.
  func.func @test_constant_of_shape_value_i64() -> tensor<3xi64> {
    // CHECK-LABEL: func.func @test_constant_of_shape_value_i64
    %shape = arith.constant dense<[3]> : tensor<1xi64>
    %r = "onnx.ConstantOfShape"(%shape) {
      value = dense<7> : tensor<1xi64>
    } : (tensor<1xi64>) -> tensor<3xi64>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK: arith.constant dense<7> : tensor<3xi64>

    return %r : tensor<3xi64>
  }

  // Note: the canonical "Shape -> ConstantOfShape" composition test
  // (transformer-style allocation of a zero KV/mask tensor) lives with
  // the Shape conversion in a separate PR. Once that PR lands, re-add a
  // composed test here.
}
