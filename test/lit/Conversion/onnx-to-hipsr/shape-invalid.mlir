// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Shape forms the conversion rejects.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// The number of extents to read comes from the input's rank.
func.func @unranked_input(%ctx: !hipsr.context, %input: tensor<*xf16>)
    -> tensor<3xi64, #hipsr.mem<host>> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input) : (tensor<*xf16>) -> tensor<3xi64, #hipsr.mem<host>>
  return %0 : tensor<3xi64, #hipsr.mem<host>>
}

// -----

// ONNX Shape returns extents as i64.
func.func @result_element_type(%ctx: !hipsr.context,
                               %input: tensor<2x3xf16>)
    -> tensor<2xi32, #hipsr.mem<host>> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input)
      : (tensor<2x3xf16>) -> tensor<2xi32, #hipsr.mem<host>>
  return %0 : tensor<2xi32, #hipsr.mem<host>>
}

// -----

// The result is an extent vector, so it must be rank 1.
func.func @result_rank(%ctx: !hipsr.context, %input: tensor<2x3xf16>)
    -> tensor<1x2xi64, #hipsr.mem<host>> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input)
      : (tensor<2x3xf16>) -> tensor<1x2xi64, #hipsr.mem<host>>
  return %0 : tensor<1x2xi64, #hipsr.mem<host>>
}

// -----

// The body fills a host buffer, so a result naming another space is rejected
// rather than copied. A result naming no space is fine; the conversion names
// host for it.
func.func @result_names_device(%ctx: !hipsr.context, %input: tensor<2x3xf16>)
    -> tensor<2xi64, #hipsr.mem<device>> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input)
      : (tensor<2x3xf16>) -> tensor<2xi64, #hipsr.mem<device>>
  return %0 : tensor<2xi64, #hipsr.mem<device>>
}

// -----

// ONNX allows an empty axis range, but a zero-extent destination has nothing to
// allocate.
func.func @empty_axis_range(%ctx: !hipsr.context, %input: tensor<2x3x4xf16>)
    -> tensor<0xi64, #hipsr.mem<host>> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input) {start = 2 : si64, end = 1 : si64}
      : (tensor<2x3x4xf16>) -> tensor<0xi64, #hipsr.mem<host>>
  return %0 : tensor<0xi64, #hipsr.mem<host>>
}

// -----

// The result length must equal the number of selected axes.
func.func @result_length(%ctx: !hipsr.context, %input: tensor<2x3x4xf16>)
    -> tensor<2xi64, #hipsr.mem<host>> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input)
      : (tensor<2x3x4xf16>) -> tensor<2xi64, #hipsr.mem<host>>
  return %0 : tensor<2xi64, #hipsr.mem<host>>
}

// -----

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%input: tensor<2x3xf16>)
    -> tensor<2xi64, #hipsr.mem<host>> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Shape'}}
  %0 = "onnx.Shape"(%input)
      : (tensor<2x3xf16>) -> tensor<2xi64, #hipsr.mem<host>>
  return %0 : tensor<2xi64, #hipsr.mem<host>>
}
