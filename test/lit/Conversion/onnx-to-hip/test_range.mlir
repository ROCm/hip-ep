// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata in convert-onnx-to-hip.
  func.func @main_graph(%arg0: tensor<i32>) -> tensor<i32> {
    return %arg0 : tensor<i32>
  }

  // Scalar constant operands still lower through hip.range (no compile-time fold).
  func.func @test_range_i32_constants() -> tensor<4xi32> {
    %s = arith.constant dense<2> : tensor<i32>
    %l = arith.constant dense<10> : tensor<i32>
    %d = arith.constant dense<2> : tensor<i32>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
  // CHECK-LABEL: func.func @test_range_i32_constants
  // CHECK-SAME: !hip.context
  // CHECK: tensor.empty
  // CHECK: hip.range

  // Dynamic operands lower to hip.range with DPS init tensor.
  func.func @test_range_i32_dynamic(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?xi32> {
    %r = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
    return %r : tensor<?xi32>
  }
  // CHECK-LABEL: func.func @test_range_i32_dynamic
  // CHECK: tensor.empty
  // CHECK: hip.range

  func.func @test_range_i16() -> tensor<4xi16> {
    %s = arith.constant dense<0> : tensor<i16>
    %l = arith.constant dense<4> : tensor<i16>
    %d = arith.constant dense<1> : tensor<i16>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<i16>, tensor<i16>, tensor<i16>) -> tensor<4xi16>
    return %r : tensor<4xi16>
  }
  // CHECK-LABEL: func.func @test_range_i16
  // CHECK: hip.range

  func.func @test_range_i64() -> tensor<4xi64> {
    %s = arith.constant dense<0> : tensor<i64>
    %l = arith.constant dense<4> : tensor<i64>
    %d = arith.constant dense<1> : tensor<i64>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4xi64>
    return %r : tensor<4xi64>
  }
  // CHECK-LABEL: func.func @test_range_i64
  // CHECK: hip.range

  func.func @test_range_f64() -> tensor<4xf64> {
    %s = arith.constant dense<0.0> : tensor<f64>
    %l = arith.constant dense<4.0> : tensor<f64>
    %d = arith.constant dense<1.0> : tensor<f64>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<4xf64>
    return %r : tensor<4xf64>
  }
  // CHECK-LABEL: func.func @test_range_f64
  // CHECK: hip.range

  // Empty ranges are still lowered to hip.range. Runtime gets an empty output.
  func.func @test_range_empty_pos_i32() -> tensor<0xi32> {
    %s = arith.constant dense<5> : tensor<i32>
    %l = arith.constant dense<2> : tensor<i32>
    %d = arith.constant dense<1> : tensor<i32>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<0xi32>
    return %r : tensor<0xi32>
  }
  // CHECK-LABEL: func.func @test_range_empty_pos_i32
  // CHECK: tensor.empty
  // CHECK: hip.range

  // Negative delta with increasing interval also yields empty output.
  func.func @test_range_empty_neg_f32() -> tensor<0xf32> {
    %s = arith.constant dense<0.0> : tensor<f32>
    %l = arith.constant dense<5.0> : tensor<f32>
    %d = arith.constant dense<-1.0> : tensor<f32>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<0xf32>
    return %r : tensor<0xf32>
  }
  // CHECK-LABEL: func.func @test_range_empty_neg_f32
  // CHECK: tensor.empty
  // CHECK: hip.range

  // A single-element rank-1 bound (HF exports slice a length out of Shape as
  // tensor<1xi64>) is accepted and collapsed to rank-0 before the readback +
  // hip.range path. The dynamic limit reads back through a synchronized scalar.
  func.func @test_range_rank1_limit(%limit: tensor<1xi64>) -> tensor<?xi64> {
    %s = arith.constant dense<0> : tensor<i64>
    %d = arith.constant dense<1> : tensor<i64>
    %r = "onnx.Range"(%s, %limit, %d) : (tensor<i64>, tensor<1xi64>, tensor<i64>) -> tensor<?xi64>
    return %r : tensor<?xi64>
  }
  // CHECK-LABEL: func.func @test_range_rank1_limit
  // CHECK: tensor.collapse_shape %{{.*}} [] : tensor<1xi64> into tensor<i64>
  // CHECK: hip.readback_scalar
  // CHECK: hip.range
}
