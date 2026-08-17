// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics

module {
  func.func @main_graph(%arg0: tensor<?x?x512xf32>,
                        %axes: tensor<i64>) -> tensor<?x?xf32> {
    // expected-error @+1 {{reduction axes must be known at compile time}}
    %output = "onnx.ReduceSum"(%arg0, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<?x?x512xf32>, tensor<i64>) -> tensor<?x?xf32>
    return %output : tensor<?x?xf32>
  }

  func.func @runtime_axes_static_result(%data: tensor<2x3xf32>,
                                        %axes: tensor<1xi64>)
      -> tensor<2xf32> {
    // expected-error @+1 {{reduction axes must be known at compile time}}
    %output = "onnx.ReduceProd"(%data, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2xf32>
    return %output : tensor<2xf32>
  }

  func.func @noncontiguous_axes(%data: tensor<2x3x4xf32>)
      -> tensor<3xf32> {
    %axes = arith.constant dense<[0, 2]> : tensor<2xi64>
    // expected-error @+1 {{reduction axes must be unique, in range, and form one contiguous span}}
    %output = "onnx.ReduceMean"(%data, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<3xf32>
    return %output : tensor<3xf32>
  }

  func.func @constant_carrier_shape_contradiction(
      %data: tensor<2x3x4xf32>) -> tensor<2x5xf32> {
    %axes = "onnx.Constant"() {
      value = dense<[1]> : tensor<1xi64>
    } : () -> tensor<1xi64>
    // expected-error @+1 {{result type is incompatible with the reduction data shape and axes}}
    %output = "onnx.ReduceL2"(%data, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x5xf32>
    return %output : tensor<2x5xf32>
  }
}
