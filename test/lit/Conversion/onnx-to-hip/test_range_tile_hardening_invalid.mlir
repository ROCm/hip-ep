// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip \
// RUN:   --split-input-file --verify-diagnostics

module {
  func.func @main_graph() -> tensor<?xi64> {
    %start = arith.constant dense<0> : tensor<i64>
    %limit = arith.constant dense<4> : tensor<i64>
    %delta = arith.constant dense<0> : tensor<i64>
    // expected-error@+1 {{delta in Range operator can not be zero}}
    %result = "onnx.Range"(%start, %limit, %delta)
      : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
    return %result : tensor<?xi64>
  }
}

// -----

module {
  func.func @main_graph() -> tensor<?xf32> {
    %start = arith.constant dense<0.0> : tensor<f32>
    %limit = arith.constant dense<0x7FC00000> : tensor<f32>
    %delta = arith.constant dense<1.0> : tensor<f32>
    // expected-error@+1 {{constant Range controls produce an invalid or unrepresentable result length}}
    %result = "onnx.Range"(%start, %limit, %delta)
      : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
    return %result : tensor<?xf32>
  }
}

// -----

module {
  func.func @main_graph() -> tensor<?xf32> {
    %start = arith.constant dense<0.0> : tensor<f32>
    %limit = arith.constant dense<0x7F800000> : tensor<f32>
    %delta = arith.constant dense<1.0> : tensor<f32>
    // expected-error@+1 {{constant Range controls produce an invalid or unrepresentable result length}}
    %result = "onnx.Range"(%start, %limit, %delta)
      : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
    return %result : tensor<?xf32>
  }
}

// -----

module {
  func.func @main_graph() -> tensor<?xi64> {
    %start = arith.constant dense<-9223372036854775808> : tensor<i64>
    %limit = arith.constant dense<9223372036854775807> : tensor<i64>
    %delta = arith.constant dense<1> : tensor<i64>
    // expected-error@+1 {{constant Range controls produce an invalid or unrepresentable result length}}
    %result = "onnx.Range"(%start, %limit, %delta)
      : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
    return %result : tensor<?xi64>
  }
}

// -----

module {
  func.func @main_graph(%input: tensor<2xf32>) -> tensor<?xf32> {
    %repeats = arith.constant dense<[-1]> : tensor<1xi64>
    // expected-error@+1 {{constant Tile repeats or extent products are invalid or unrepresentable}}
    %result = "onnx.Tile"(%input, %repeats)
      : (tensor<2xf32>, tensor<1xi64>) -> tensor<?xf32>
    return %result : tensor<?xf32>
  }
}

// -----

module {
  func.func @main_graph(
      %input: tensor<9223372036854775807xf32>) -> tensor<?xf32> {
    %repeats = arith.constant dense<[2]> : tensor<1xi64>
    // expected-error@+1 {{constant Tile repeats or extent products are invalid or unrepresentable}}
    %result = "onnx.Tile"(%input, %repeats)
      : (tensor<9223372036854775807xf32>, tensor<1xi64>) -> tensor<?xf32>
    return %result : tensor<?xf32>
  }
}
