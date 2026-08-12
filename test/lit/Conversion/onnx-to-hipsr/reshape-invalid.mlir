// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Reshape forms the conversion rejects.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

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

// Two dynamic result extents cannot both follow from the element count.
func.func @two_dynamic_results(%ctx: !hipsr.context,
                               %input: tensor<?x?x4xf16>,
                               %shape: tensor<2xi64>) -> tensor<?x?xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<?x?x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// Equal ranks are either the identity, handled without a destination, or a
// permutation of extents, which is not a metadata change.
func.func @equal_rank(%ctx: !hipsr.context, %input: tensor<4x3xf16>,
                      %shape: tensor<2xi64>) -> tensor<2x6xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<4x3xf16>, tensor<2xi64>) -> tensor<2x6xf16>
  return %0 : tensor<2x6xf16>
}

// -----

// No grouping of 2x3x4 yields a leading extent of 8, so the reshape moves
// data rather than regrouping extents.
func.func @no_reassociation(%ctx: !hipsr.context, %input: tensor<2x3x4xf16>,
                            %shape: tensor<2xi64>) -> tensor<8x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<2x3x4xf16>, tensor<2xi64>) -> tensor<8x3xf16>
  return %0 : tensor<8x3xf16>
}

// -----

// The reassociation is derived from the extents, so the input must be ranked.
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

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%input: tensor<2x3xf16>,
                           %shape: tensor<1xi64>) -> tensor<6xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<2x3xf16>, tensor<1xi64>) -> tensor<6xf16>
  return %0 : tensor<6xf16>
}
