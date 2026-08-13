// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Shape forms the conversion rejects.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// The number of extents to read comes from the input's rank.
func.func @unranked_input(%ctx: !hipsr.context,
                          %input: tensor<*xf16>) -> tensor<3xi64> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input) : (tensor<*xf16>) -> tensor<3xi64>
  return %0 : tensor<3xi64>
}

// -----

// ONNX Shape returns extents as i64.
func.func @result_element_type(%ctx: !hipsr.context,
                               %input: tensor<2x3xf16>) -> tensor<2xi32> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input) : (tensor<2x3xf16>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// The result is an extent vector, so it must be rank 1.
func.func @result_rank(%ctx: !hipsr.context,
                       %input: tensor<2x3xf16>) -> tensor<1x2xi64> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input) : (tensor<2x3xf16>) -> tensor<1x2xi64>
  return %0 : tensor<1x2xi64>
}

// -----

// ONNX allows an empty axis range, but a zero-extent destination has nothing
// to allocate, so the conversion reports it instead.
func.func @empty_axis_range(%ctx: !hipsr.context,
                            %input: tensor<2x3x4xf16>) -> tensor<0xi64> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input) {start = 2 : si64, end = 1 : si64}
      : (tensor<2x3x4xf16>) -> tensor<0xi64>
  return %0 : tensor<0xi64>
}

// -----

// The result length must equal the number of selected axes.
func.func @result_length(%ctx: !hipsr.context,
                         %input: tensor<2x3x4xf16>) -> tensor<2xi64> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input) : (tensor<2x3x4xf16>) -> tensor<2xi64>
  return %0 : tensor<2xi64>
}

// -----

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%input: tensor<2x3xf16>) -> tensor<2xi64> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input) : (tensor<2x3xf16>) -> tensor<2xi64>
  return %0 : tensor<2xi64>
}
