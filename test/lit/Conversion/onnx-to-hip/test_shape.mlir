// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify that onnx.Shape is folded to an arith.constant at compile time.
// Because the compiler requires fully static input shapes, the result is
// always known and no runtime support is required.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: default attrs (start=0, end=rank). Output values = full input shape.
  func.func @test_shape_default(%input: tensor<2x3x4xf32>) -> tensor<3xi64> {
    // CHECK-LABEL: func.func @test_shape_default
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<2x3x4xf32>) -> tensor<3xi64>

    %r = "onnx.Shape"(%input) : (tensor<2x3x4xf32>) -> tensor<3xi64>

    // CHECK-NOT: onnx.Shape
    // CHECK: arith.constant dense<[2, 3, 4]> : tensor<3xi64>

    return %r : tensor<3xi64>
  }

  // Test 2: explicit start and end (slice middle dims).
  func.func @test_shape_start_end(%input: tensor<2x3x4x5xf32>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_shape_start_end
    %r = "onnx.Shape"(%input) {start = 1 : si64, end = 3 : si64}
        : (tensor<2x3x4x5xf32>) -> tensor<2xi64>

    // CHECK-NOT: onnx.Shape
    // CHECK: arith.constant dense<[3, 4]> : tensor<2xi64>

    return %r : tensor<2xi64>
  }

  // Test 3: negative end (-1 means "up to but not including last axis").
  func.func @test_shape_negative_end(%input: tensor<2x3x4xf32>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_shape_negative_end
    %r = "onnx.Shape"(%input) {end = -1 : si64}
        : (tensor<2x3x4xf32>) -> tensor<2xi64>

    // CHECK-NOT: onnx.Shape
    // CHECK: arith.constant dense<[2, 3]> : tensor<2xi64>

    return %r : tensor<2xi64>
  }

  // Test 4: negative start (counts from the back).
  func.func @test_shape_negative_start(%input: tensor<2x3x4xf32>) -> tensor<1xi64> {
    // CHECK-LABEL: func.func @test_shape_negative_start
    %r = "onnx.Shape"(%input) {start = -1 : si64}
        : (tensor<2x3x4xf32>) -> tensor<1xi64>

    // CHECK-NOT: onnx.Shape
    // CHECK: arith.constant dense<4> : tensor<1xi64>

    return %r : tensor<1xi64>
  }

  // Test 5: start > end yields an empty result (clamped).
  func.func @test_shape_empty(%input: tensor<2x3x4xf32>) -> tensor<0xi64> {
    // CHECK-LABEL: func.func @test_shape_empty
    %r = "onnx.Shape"(%input) {start = 2 : si64, end = 1 : si64}
        : (tensor<2x3x4xf32>) -> tensor<0xi64>

    // CHECK-NOT: onnx.Shape
    // CHECK: arith.constant dense<> : tensor<0xi64>

    return %r : tensor<0xi64>
  }
}
