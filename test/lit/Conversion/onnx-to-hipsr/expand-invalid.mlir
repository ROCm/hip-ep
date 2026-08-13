// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Expand forms the conversion rejects. A pattern that does not match
// leaves the ONNX operation in place, and the strict conversion target then
// fails the pass, so each case expects a legalization error.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// The shape operand is an extent vector, so it must be rank 1.
func.func @shape_rank(%ctx: !hipsr.context, %input: tensor<2x3xf16>,
                      %shape: tensor<1x2xi64>) -> tensor<2x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Expand'}}
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<1x2xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// ONNX shape extents are i64.
func.func @shape_element_type(%ctx: !hipsr.context, %input: tensor<2x3xf16>,
                              %shape: tensor<2xi32>) -> tensor<2x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Expand'}}
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<2xi32>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// The shape length fixes the output rank, so it cannot itself be dynamic.
func.func @dynamic_shape_length(%ctx: !hipsr.context, %input: tensor<2x3xf16>,
                                %shape: tensor<?xi64>) -> tensor<2x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Expand'}}
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<?xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// An unranked input has no rank to broadcast against.
func.func @unranked_input(%ctx: !hipsr.context, %input: tensor<*xf16>,
                          %shape: tensor<2xi64>) -> tensor<2x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Expand'}}
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<*xf16>, tensor<2xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// Expand only broadcasts, so it cannot change the element type.
func.func @element_type_mismatch(%ctx: !hipsr.context, %input: tensor<2x3xf16>,
                                 %shape: tensor<2xi64>) -> tensor<2x3xf32> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Expand'}}
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<2xi64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// The result rank must be max(input rank, shape length); 2 is neither here.
func.func @result_rank(%ctx: !hipsr.context, %input: tensor<2x3xf16>,
                       %shape: tensor<4xi64>) -> tensor<2x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Expand'}}
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<4xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%input: tensor<2x3xf16>,
                           %shape: tensor<2xi64>) -> tensor<2x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Expand'}}
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<2xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}
