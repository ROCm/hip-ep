// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics

// Operands of different element types cannot be compared.
func.func @equal_operand_element_mismatch(
    %ctx: !hipsr.context, %lhs: tensor<2x3xf16, #hipsr.mem<device>>,
    %rhs: tensor<2x3xf32, #hipsr.mem<device>>,
    %init: tensor<2x3xui8, #hipsr.mem<device>>)
    -> tensor<2x3xui8, #hipsr.mem<device>> {
  // expected-error@+1 {{failed to verify that all of {lhs, rhs} have same element type}}
  %result = hipsr.equal(%ctx)
      ins(%lhs, %rhs : tensor<2x3xf16, #hipsr.mem<device>>,
                       tensor<2x3xf32, #hipsr.mem<device>>)
      outs(%init : tensor<2x3xui8, #hipsr.mem<device>>)
      : tensor<2x3xui8, #hipsr.mem<device>>
  return %result : tensor<2x3xui8, #hipsr.mem<device>>
}

// -----

// The mask is the byte the runtime writes. The conversion turns the bit ONNX
// declares into ui8 before building the op.
func.func @equal_bit_mask(
    %ctx: !hipsr.context, %lhs: tensor<2x3xf16, #hipsr.mem<device>>,
    %rhs: tensor<2x3xf16, #hipsr.mem<device>>,
    %init: tensor<2x3xi1, #hipsr.mem<device>>)
    -> tensor<2x3xi1, #hipsr.mem<device>> {
  // expected-error@+1 {{output element type must be ui8}}
  %result = hipsr.equal(%ctx)
      ins(%lhs, %rhs : tensor<2x3xf16, #hipsr.mem<device>>,
                       tensor<2x3xf16, #hipsr.mem<device>>)
      outs(%init : tensor<2x3xi1, #hipsr.mem<device>>)
      : tensor<2x3xi1, #hipsr.mem<device>>
  return %result : tensor<2x3xi1, #hipsr.mem<device>>
}

// -----

// Aligned extents 3 and 4 are neither equal nor one.
func.func @equal_incompatible_operands(
    %ctx: !hipsr.context, %lhs: tensor<2x3xf16, #hipsr.mem<device>>,
    %rhs: tensor<2x4xf16, #hipsr.mem<device>>,
    %init: tensor<2x3xui8, #hipsr.mem<device>>)
    -> tensor<2x3xui8, #hipsr.mem<device>> {
  // expected-error@+1 {{lhs and rhs shapes are not broadcast compatible}}
  %result = hipsr.equal(%ctx)
      ins(%lhs, %rhs : tensor<2x3xf16, #hipsr.mem<device>>,
                       tensor<2x4xf16, #hipsr.mem<device>>)
      outs(%init : tensor<2x3xui8, #hipsr.mem<device>>)
      : tensor<2x3xui8, #hipsr.mem<device>>
  return %result : tensor<2x3xui8, #hipsr.mem<device>>
}

// -----

// The output holds the broadcast of both operand shapes, here 4x1 with 3.
func.func @equal_output_shape(
    %ctx: !hipsr.context, %lhs: tensor<4x1xf16, #hipsr.mem<device>>,
    %rhs: tensor<3xf16, #hipsr.mem<device>>,
    %init: tensor<4x1xui8, #hipsr.mem<device>>)
    -> tensor<4x1xui8, #hipsr.mem<device>> {
  // expected-error@+1 {{output shape must be the broadcast of lhs and rhs}}
  %result = hipsr.equal(%ctx)
      ins(%lhs, %rhs : tensor<4x1xf16, #hipsr.mem<device>>,
                       tensor<3xf16, #hipsr.mem<device>>)
      outs(%init : tensor<4x1xui8, #hipsr.mem<device>>)
      : tensor<4x1xui8, #hipsr.mem<device>>
  return %result : tensor<4x1xui8, #hipsr.mem<device>>
}
