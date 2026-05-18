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

  // Test 4: shape comes from an onnx.Constant (folded by the same pipeline
  // run because the Shape and ConstantOfShape folds compose under the greedy
  // rewriter). This mirrors the canonical "Shape -> ConstantOfShape" pattern
  // emitted by transformers when allocating a zero KV/mask tensor.
  func.func @test_shape_then_constant_of_shape(%data: tensor<2x4xf32>) -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @test_shape_then_constant_of_shape
    %shape = "onnx.Shape"(%data) : (tensor<2x4xf32>) -> tensor<2xi64>
    %zeros = "onnx.ConstantOfShape"(%shape) : (tensor<2xi64>) -> tensor<2x4xf32>

    // CHECK-NOT: onnx.Shape
    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK: arith.constant dense<0.000000e+00> : tensor<2x4xf32>

    return %zeros : tensor<2x4xf32>
  }
}
