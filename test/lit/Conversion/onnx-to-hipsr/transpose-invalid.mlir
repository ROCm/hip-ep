// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Transpose forms the conversion rejects.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// perm names one destination axis per input axis.
func.func @perm_length(%ctx: !hipsr.context,
                       %input: tensor<2x3x4xf16>) -> tensor<4x3x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Transpose'}}
  %0 = "onnx.Transpose"(%input) {perm = [1, 0]}
      : (tensor<2x3x4xf16>) -> tensor<4x3x2xf16>
  return %0 : tensor<4x3x2xf16>
}

// -----

// Repeating an axis would drop another one.
func.func @perm_repeated(%ctx: !hipsr.context,
                         %input: tensor<2x3xf16>) -> tensor<2x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Transpose'}}
  %0 = "onnx.Transpose"(%input) {perm = [0, 0]}
      : (tensor<2x3xf16>) -> tensor<2x2xf16>
  return %0 : tensor<2x2xf16>
}

// -----

// perm entries are axis indices.
func.func @perm_not_integers(%ctx: !hipsr.context,
                             %input: tensor<2x3xf16>) -> tensor<3x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Transpose'}}
  %0 = "onnx.Transpose"(%input) {perm = ["a", "b"]}
      : (tensor<2x3xf16>) -> tensor<3x2xf16>
  return %0 : tensor<3x2xf16>
}

// -----

// A permutation is over the input's axes, so the input must be ranked.
func.func @unranked_input(%ctx: !hipsr.context,
                          %input: tensor<*xf16>) -> tensor<3x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Transpose'}}
  %0 = "onnx.Transpose"(%input) {perm = [1, 0]}
      : (tensor<*xf16>) -> tensor<3x2xf16>
  return %0 : tensor<3x2xf16>
}

// -----

// Permuting axes never changes the rank.
func.func @rank_change(%ctx: !hipsr.context,
                       %input: tensor<2x3xf16>) -> tensor<6xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Transpose'}}
  %0 = "onnx.Transpose"(%input) {perm = [1, 0]}
      : (tensor<2x3xf16>) -> tensor<6xf16>
  return %0 : tensor<6xf16>
}

// -----

// Transpose moves elements; it does not convert them.
func.func @element_type_mismatch(%ctx: !hipsr.context,
                                 %input: tensor<2x3xf16>) -> tensor<3x2xf32> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Transpose'}}
  %0 = "onnx.Transpose"(%input) {perm = [1, 0]}
      : (tensor<2x3xf16>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// -----

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%input: tensor<2x3xf16>) -> tensor<3x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Transpose'}}
  %0 = "onnx.Transpose"(%input) {perm = [1, 0]}
      : (tensor<2x3xf16>) -> tensor<3x2xf16>
  return %0 : tensor<3x2xf16>
}
