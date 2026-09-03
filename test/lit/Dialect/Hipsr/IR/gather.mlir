// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics

// ONNX normalizes a negative axis before it reaches the dialect.
func.func @gather_negative_axis(
    %ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>,
    %indices: tensor<5xi64, #hipsr.mem<device>>,
    %init: tensor<2x5xf16, #hipsr.mem<device>>)
    -> tensor<2x5xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{axis must be in [0, data rank); data rank is 2, got -1}}
  %result = hipsr.gather(%ctx)
      ins(%data, %indices : tensor<2x3xf16, #hipsr.mem<device>>,
                            tensor<5xi64, #hipsr.mem<device>>)
      outs(%init : tensor<2x5xf16, #hipsr.mem<device>>) {axis = -1 : i64}
      : tensor<2x5xf16, #hipsr.mem<device>>
  return %result : tensor<2x5xf16, #hipsr.mem<device>>
}

// -----

// Only the data's own axes can be gathered.
func.func @gather_axis_out_of_range(
    %ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>,
    %indices: tensor<5xi64, #hipsr.mem<device>>,
    %init: tensor<2x5xf16, #hipsr.mem<device>>)
    -> tensor<2x5xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{axis must be in [0, data rank); data rank is 2, got 2}}
  %result = hipsr.gather(%ctx)
      ins(%data, %indices : tensor<2x3xf16, #hipsr.mem<device>>,
                            tensor<5xi64, #hipsr.mem<device>>)
      outs(%init : tensor<2x5xf16, #hipsr.mem<device>>) {axis = 2 : i64}
      : tensor<2x5xf16, #hipsr.mem<device>>
  return %result : tensor<2x5xf16, #hipsr.mem<device>>
}

// -----

// Indices name positions, so they cannot be floating point.
func.func @gather_float_indices(
    %ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>,
    %indices: tensor<5xf32, #hipsr.mem<device>>,
    %init: tensor<5x3xf16, #hipsr.mem<device>>)
    -> tensor<5x3xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{indices element type must be an integer}}
  %result = hipsr.gather(%ctx)
      ins(%data, %indices : tensor<2x3xf16, #hipsr.mem<device>>,
                            tensor<5xf32, #hipsr.mem<device>>)
      outs(%init : tensor<5x3xf16, #hipsr.mem<device>>) {axis = 0 : i64}
      : tensor<5x3xf16, #hipsr.mem<device>>
  return %result : tensor<5x3xf16, #hipsr.mem<device>>
}

// -----

// Gather selects elements; it does not convert them.
func.func @gather_element_mismatch(
    %ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>,
    %indices: tensor<5xi64, #hipsr.mem<device>>,
    %init: tensor<5x3xf32, #hipsr.mem<device>>)
    -> tensor<5x3xf32, #hipsr.mem<device>> {
  // expected-error@+1 {{failed to verify that all of {data, init} have same element type}}
  %result = hipsr.gather(%ctx)
      ins(%data, %indices : tensor<2x3xf16, #hipsr.mem<device>>,
                            tensor<5xi64, #hipsr.mem<device>>)
      outs(%init : tensor<5x3xf32, #hipsr.mem<device>>) {axis = 0 : i64}
      : tensor<5x3xf32, #hipsr.mem<device>>
  return %result : tensor<5x3xf32, #hipsr.mem<device>>
}

// -----

// The gathered axis is replaced by the whole indices shape, so the output here
// must be 5x3 rather than the data's own 2x3.
func.func @gather_output_shape(
    %ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>,
    %indices: tensor<5xi64, #hipsr.mem<device>>,
    %init: tensor<2x3xf16, #hipsr.mem<device>>)
    -> tensor<2x3xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{output shape must be the data shape with axis replaced by the indices shape}}
  %result = hipsr.gather(%ctx)
      ins(%data, %indices : tensor<2x3xf16, #hipsr.mem<device>>,
                            tensor<5xi64, #hipsr.mem<device>>)
      outs(%init : tensor<2x3xf16, #hipsr.mem<device>>) {axis = 0 : i64}
      : tensor<2x3xf16, #hipsr.mem<device>>
  return %result : tensor<2x3xf16, #hipsr.mem<device>>
}
