// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Unsqueeze forms the conversion rejects for good: axes ONNX calls an
// error, and an unranked input. Forms that are only unimplemented are not here;
// TODOs mark those in the conversion.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --onnx-dialect=modeled --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// ONNX calls a repeated axis an error, and it would leave an input dimension
// with nowhere to go.
func.func @repeated_axis(%ctx: !hipsr.context,
                         %input: tensor<4xf16>) -> tensor<1x1x4xf16> {
  %axes = "onnx.Constant"() {value = dense<[0, 0]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Unsqueeze'}}
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<4xf16>, tensor<2xi64>) -> tensor<1x1x4xf16>
  return %0 : tensor<1x1x4xf16>
}

// -----

// The axes and the input rank fix the result rank at 2, so axis 2 names no
// position in it.
func.func @axis_out_of_range(%ctx: !hipsr.context,
                             %input: tensor<4xf16>) -> tensor<1x4xf16> {
  %axes = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
      : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Unsqueeze'}}
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<4xf16>, tensor<1xi64>) -> tensor<1x4xf16>
  return %0 : tensor<1x4xf16>
}

// -----

// A negative axis counts from the end of the result, which has rank 2 here, so
// -3 reaches past its start.
func.func @negative_axis_out_of_range(%ctx: !hipsr.context,
                                      %input: tensor<4xf16>) -> tensor<1x4xf16> {
  %axes = "onnx.Constant"() {value = dense<-3> : tensor<1xi64>}
      : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Unsqueeze'}}
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<4xf16>, tensor<1xi64>) -> tensor<1x4xf16>
  return %0 : tensor<1x4xf16>
}

// -----

// Every dimension the axes do not insert comes from the input, so the input has
// to be ranked.
func.func @unranked_input(%ctx: !hipsr.context,
                          %input: tensor<*xf16>) -> tensor<1x4xf16> {
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Unsqueeze'}}
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<*xf16>, tensor<1xi64>) -> tensor<1x4xf16>
  return %0 : tensor<1x4xf16>
}
