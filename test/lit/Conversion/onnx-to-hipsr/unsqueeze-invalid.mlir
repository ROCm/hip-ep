// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Unsqueeze forms the conversion rejects.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// Regrouping 8 into 2x4 moves data rather than inserting a unit axis.
func.func @regrouped_extent(%ctx: !hipsr.context, %input: tensor<4x8xf16>,
                            %axes: tensor<1xi64>) -> tensor<4x2x4xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Unsqueeze'}}
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x2x4xf16>
  return %0 : tensor<4x2x4xf16>
}

// -----

// Inserting unit axes into 8x1 cannot put the 8 last, so this reorders axes.
func.func @reordered_axes(%ctx: !hipsr.context, %input: tensor<8x1xf16>,
                          %axes: tensor<1xi64>) -> tensor<1x1x8xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Unsqueeze'}}
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<8x1xf16>, tensor<1xi64>) -> tensor<1x1x8xf16>
  return %0 : tensor<1x1x8xf16>
}

// -----

// Dropping an axis is a squeeze, not an unsqueeze.
func.func @rank_decrease(%ctx: !hipsr.context, %input: tensor<1x4xf16>,
                         %axes: tensor<1xi64>) -> tensor<4xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Unsqueeze'}}
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<1x4xf16>, tensor<1xi64>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// The grouping is derived from the extents, so the input must be ranked.
func.func @unranked_input(%ctx: !hipsr.context, %input: tensor<*xf16>,
                          %axes: tensor<1xi64>) -> tensor<1x4xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Unsqueeze'}}
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<*xf16>, tensor<1xi64>) -> tensor<1x4xf16>
  return %0 : tensor<1x4xf16>
}

// -----

// Unsqueeze inserts axes; it does not convert elements.
func.func @element_type_mismatch(%ctx: !hipsr.context,
                                 %input: tensor<4xf16>,
                                 %axes: tensor<1xi64>) -> tensor<1x4xf32> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Unsqueeze'}}
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<4xf16>, tensor<1xi64>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%input: tensor<4xf16>,
                           %axes: tensor<1xi64>) -> tensor<1x4xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Unsqueeze'}}
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<4xf16>, tensor<1xi64>) -> tensor<1x4xf16>
  return %0 : tensor<1x4xf16>
}
