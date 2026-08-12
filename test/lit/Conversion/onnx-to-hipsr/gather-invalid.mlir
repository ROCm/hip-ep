// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Gather forms the conversion rejects.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// Only the data's own axes can be gathered, before or after normalization.
func.func @axis_out_of_range(%ctx: !hipsr.context, %data: tensor<2x3xf16>,
                             %indices: tensor<5xi64>) -> tensor<2x5xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Gather'}}
  %0 = "onnx.Gather"(%data, %indices) {axis = 2 : si64}
      : (tensor<2x3xf16>, tensor<5xi64>) -> tensor<2x5xf16>
  return %0 : tensor<2x5xf16>
}

// -----

// A negative axis below -rank has no axis to normalize to.
func.func @axis_below_negative_rank(%ctx: !hipsr.context,
                                    %data: tensor<2x3xf16>,
                                    %indices: tensor<5xi64>)
    -> tensor<5x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Gather'}}
  %0 = "onnx.Gather"(%data, %indices) {axis = -3 : si64}
      : (tensor<2x3xf16>, tensor<5xi64>) -> tensor<5x3xf16>
  return %0 : tensor<5x3xf16>
}

// -----

// Indices name positions, so they cannot be floating point.
func.func @float_indices(%ctx: !hipsr.context, %data: tensor<2x3xf16>,
                         %indices: tensor<5xf32>) -> tensor<5x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Gather'}}
  %0 = "onnx.Gather"(%data, %indices) {axis = 0 : si64}
      : (tensor<2x3xf16>, tensor<5xf32>) -> tensor<5x3xf16>
  return %0 : tensor<5x3xf16>
}

// -----

// The result rank follows from both operand ranks, so both must be ranked.
func.func @unranked_data(%ctx: !hipsr.context, %data: tensor<*xf16>,
                         %indices: tensor<5xi64>) -> tensor<5x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Gather'}}
  %0 = "onnx.Gather"(%data, %indices) {axis = 0 : si64}
      : (tensor<*xf16>, tensor<5xi64>) -> tensor<5x3xf16>
  return %0 : tensor<5x3xf16>
}

// -----

// Gathering with rank-1 indices keeps the data's rank, not one less.
func.func @result_rank(%ctx: !hipsr.context, %data: tensor<2x3xf16>,
                       %indices: tensor<5xi64>) -> tensor<5xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Gather'}}
  %0 = "onnx.Gather"(%data, %indices) {axis = 0 : si64}
      : (tensor<2x3xf16>, tensor<5xi64>) -> tensor<5xf16>
  return %0 : tensor<5xf16>
}

// -----

// Gather selects elements; it does not convert them.
func.func @element_type_mismatch(%ctx: !hipsr.context, %data: tensor<2x3xf16>,
                                 %indices: tensor<5xi64>) -> tensor<5x3xf32> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Gather'}}
  %0 = "onnx.Gather"(%data, %indices) {axis = 0 : si64}
      : (tensor<2x3xf16>, tensor<5xi64>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
}

// -----

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%data: tensor<2x3xf16>,
                           %indices: tensor<5xi64>) -> tensor<5x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Gather'}}
  %0 = "onnx.Gather"(%data, %indices) {axis = 0 : si64}
      : (tensor<2x3xf16>, tensor<5xi64>) -> tensor<5x3xf16>
  return %0 : tensor<5x3xf16>
}
