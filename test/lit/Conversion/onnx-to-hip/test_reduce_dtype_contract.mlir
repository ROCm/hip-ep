// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics

module {
  func.func @main_graph(%data: tensor<2x3xf32>) -> tensor<2xf32> {
    %axes = arith.constant dense<[1]> : tensor<1xi64>
    %result = "onnx.ReduceSum"(%data, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }

  func.func @reduce_mean_f32(%data: tensor<2x3xf32>) -> tensor<2xf32> {
    %axes = arith.constant dense<[1]> : tensor<1xi64>
    %result = "onnx.ReduceMean"(%data, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }

  func.func @reduce_l2_f32(%data: tensor<2x3xf32>) -> tensor<2xf32> {
    %axes = arith.constant dense<[1]> : tensor<1xi64>
    %result = "onnx.ReduceL2"(%data, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }

  func.func @reduce_max_f32(%data: tensor<2x3xf32>) -> tensor<2xf32> {
    %axes = arith.constant dense<[1]> : tensor<1xi64>
    // expected-error @+1 {{unsupported reduction element type 'f32'; supported types: f16, i32, i64}}
    %result = "onnx.ReduceMax"(%data, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }

  func.func @reduce_min_bf16(%data: tensor<2x3xbf16>) -> tensor<2xbf16> {
    %axes = arith.constant dense<[1]> : tensor<1xi64>
    // expected-error @+1 {{unsupported reduction element type 'bf16'; supported types: f16, i32, i64}}
    %result = "onnx.ReduceMin"(%data, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<2x3xbf16>, tensor<1xi64>) -> tensor<2xbf16>
    return %result : tensor<2xbf16>
  }

  func.func @reduce_prod_f64(%data: tensor<2x3xf64>) -> tensor<2xf64> {
    %axes = arith.constant dense<[1]> : tensor<1xi64>
    // expected-error @+1 {{unsupported reduction element type 'f64'; supported types: f16, i32, i64}}
    %result = "onnx.ReduceProd"(%data, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<2x3xf64>, tensor<1xi64>) -> tensor<2xf64>
    return %result : tensor<2xf64>
  }
}
