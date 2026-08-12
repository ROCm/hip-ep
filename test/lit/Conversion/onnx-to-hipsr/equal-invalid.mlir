// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Equal forms the conversion rejects.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// Comparing values of different types has no defined meaning.
func.func @operand_element_mismatch(%ctx: !hipsr.context,
                                    %lhs: tensor<2x3xf16>,
                                    %rhs: tensor<2x3xf32>) -> tensor<2x3xui8> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Equal'}}
  %0 = "onnx.Equal"(%lhs, %rhs)
      : (tensor<2x3xf16>, tensor<2x3xf32>) -> tensor<2x3xui8>
  return %0 : tensor<2x3xui8>
}

// -----

// The result is a mask, not a value of the operands' type.
func.func @result_element_type(%ctx: !hipsr.context, %lhs: tensor<2x3xf16>,
                               %rhs: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Equal'}}
  %0 = "onnx.Equal"(%lhs, %rhs)
      : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// The destination's rank comes from the operand ranks, so both must be ranked.
func.func @unranked_operand(%ctx: !hipsr.context, %lhs: tensor<*xf16>,
                            %rhs: tensor<2x3xf16>) -> tensor<2x3xui8> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Equal'}}
  %0 = "onnx.Equal"(%lhs, %rhs)
      : (tensor<*xf16>, tensor<2x3xf16>) -> tensor<2x3xui8>
  return %0 : tensor<2x3xui8>
}

// -----

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%lhs: tensor<2x3xf16>,
                           %rhs: tensor<2x3xf16>) -> tensor<2x3xui8> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Equal'}}
  %0 = "onnx.Equal"(%lhs, %rhs)
      : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xui8>
  return %0 : tensor<2x3xui8>
}
