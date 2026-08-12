// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Reshape forms the conversion rejects.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --onnx-dialect=modeled --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// With allowzero set, a 0 in the target shape is a literal extent instead of a
// copy of the input's, which the result type alone does not distinguish.
func.func @allowzero(%ctx: !hipsr.context, %input: tensor<2x3xf16>,
                     %shape: tensor<1xi64>) -> tensor<6xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 1 : si64}
      : (tensor<2x3xf16>, tensor<1xi64>) -> tensor<6xf16>
  return %0 : tensor<6xf16>
}

// -----

// Two unnamed extents are one equation with two unknowns, so they have to come
// from the target shape operand. This one is in device memory, which the shape
// region cannot read and nothing here copies to the host. @constant_shape and
// @host_shape in reshape.mlir are the same reshape with operands that do work.
func.func @device_shape(%ctx: !hipsr.context, %input: tensor<?x?x4xf16>,
                        %shape: tensor<2xi64>) -> tensor<?x?xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<?x?x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// The shape region reads one entry per result axis, so a host operand without
// one for each cannot be read that way.
func.func @short_host_shape(%ctx: !hipsr.context, %input: tensor<?x?x4xf16>,
                            %shape: tensor<1xi64, #hipsr.mem<host>>)
    -> tensor<?x?xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<?x?x4xf16>, tensor<1xi64, #hipsr.mem<host>>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// Entries are read one index at a time, so the host operand has to be 1-D.
func.func @host_shape_rank2(%ctx: !hipsr.context, %input: tensor<?x?x4xf16>,
                            %shape: tensor<1x2xi64, #hipsr.mem<host>>)
    -> tensor<?x?xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<?x?x4xf16>, tensor<1x2xi64, #hipsr.mem<host>>)
      -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// A reshape preserves the element count, so 6 elements cannot become 7 by
// regrouping them.
func.func @count_mismatch(%ctx: !hipsr.context, %input: tensor<2x3xf16>,
                          %shape: tensor<1xi64>) -> tensor<7xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<2x3xf16>, tensor<1xi64>) -> tensor<7xf16>
  return %0 : tensor<7xf16>
}

// -----

// The 1-D form is built from the input's extents, so the input must be ranked.
func.func @unranked_input(%ctx: !hipsr.context, %input: tensor<*xf16>,
                          %shape: tensor<1xi64>) -> tensor<6xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<*xf16>, tensor<1xi64>) -> tensor<6xf16>
  return %0 : tensor<6xf16>
}

// -----

// Reshape reinterprets extents; it does not convert elements.
func.func @element_type_mismatch(%ctx: !hipsr.context,
                                 %input: tensor<2x3xf16>,
                                 %shape: tensor<1xi64>) -> tensor<6xf32> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<2x3xf16>, tensor<1xi64>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// -----

// The result aliases the input, so it cannot cross memory spaces. A result
// naming a space the input does not have would need a copy.
func.func @space_mismatch(%ctx: !hipsr.context, %input: tensor<2x3xi64>,
                          %shape: tensor<1xi64>) {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<2x3xi64>, tensor<1xi64>) -> tensor<6xi64, #hipsr.mem<host>>
  return
}

// -----

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%input: tensor<2x3xf16>,
                           %shape: tensor<1xi64>) -> tensor<6xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<2x3xf16>, tensor<1xi64>) -> tensor<6xf16>
  return %0 : tensor<6xf16>
}
