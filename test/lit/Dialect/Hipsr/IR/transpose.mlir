// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics

// Transpose moves elements; it does not convert them.
func.func @transpose_element_mismatch(
    %ctx: !hipsr.context, %input: tensor<2x3xf16, #hipsr.mem<device>>,
    %init: tensor<3x2xf32, #hipsr.mem<device>>)
    -> tensor<3x2xf32, #hipsr.mem<device>> {
  // expected-error@+1 {{failed to verify that all of {input, init} have same element type}}
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<2x3xf16, #hipsr.mem<device>>)
      outs(%init : tensor<3x2xf32, #hipsr.mem<device>>)
      {perm = array<i64: 1, 0>} : tensor<3x2xf32, #hipsr.mem<device>>
  return %result : tensor<3x2xf32, #hipsr.mem<device>>
}

// -----

// perm names one destination axis per input axis.
func.func @transpose_perm_length(
    %ctx: !hipsr.context, %input: tensor<2x3x4xf16, #hipsr.mem<device>>,
    %init: tensor<4x3x2xf16, #hipsr.mem<device>>)
    -> tensor<4x3x2xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{perm must have one entry per input axis; expected 3, got 2}}
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<2x3x4xf16, #hipsr.mem<device>>)
      outs(%init : tensor<4x3x2xf16, #hipsr.mem<device>>)
      {perm = array<i64: 1, 0>} : tensor<4x3x2xf16, #hipsr.mem<device>>
  return %result : tensor<4x3x2xf16, #hipsr.mem<device>>
}

// -----

// Repeating an axis would drop another one.
func.func @transpose_perm_repeated(
    %ctx: !hipsr.context, %input: tensor<2x3xf16, #hipsr.mem<device>>,
    %init: tensor<2x2xf16, #hipsr.mem<device>>)
    -> tensor<2x2xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{perm must be a permutation of [0, rank)}}
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<2x3xf16, #hipsr.mem<device>>)
      outs(%init : tensor<2x2xf16, #hipsr.mem<device>>)
      {perm = array<i64: 0, 0>} : tensor<2x2xf16, #hipsr.mem<device>>
  return %result : tensor<2x2xf16, #hipsr.mem<device>>
}

// -----

// Output axis i holds input axis perm[i], so the extents must line up.
func.func @transpose_output_shape(
    %ctx: !hipsr.context, %input: tensor<2x3xf16, #hipsr.mem<device>>,
    %init: tensor<2x3xf16, #hipsr.mem<device>>)
    -> tensor<2x3xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{output shape must be the input shape permuted by perm}}
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<2x3xf16, #hipsr.mem<device>>)
      outs(%init : tensor<2x3xf16, #hipsr.mem<device>>)
      {perm = array<i64: 1, 0>} : tensor<2x3xf16, #hipsr.mem<device>>
  return %result : tensor<2x3xf16, #hipsr.mem<device>>
}
