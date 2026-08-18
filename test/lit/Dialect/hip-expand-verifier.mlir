// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --verify-diagnostics --split-input-file

func.func @valid_expand(%ctx: !hip.context, %input: tensor<1x2xf32>) {
  %valid = arith.constant true
  %shape = arith.constant dense<[3, 2]> : tensor<2xi64>
  %output = tensor.empty() : tensor<3x2xf32>
  %result = hip.expand(%ctx) valid(%valid)
    ins(%input, %shape : tensor<1x2xf32>, tensor<2xi64>)
    outs(%output : tensor<3x2xf32>) : tensor<3x2xf32>
  return
}

// -----

func.func @mismatched_element_type(
    %ctx: !hip.context, %input: tensor<2xf32>, %shape: tensor<1xi64>) {
  %valid = arith.constant true
  %output = tensor.empty() : tensor<2xi8>
  // expected-error @+1 {{input and output element types must match}}
  %result = hip.expand(%ctx) valid(%valid)
    ins(%input, %shape : tensor<2xf32>, tensor<1xi64>)
    outs(%output : tensor<2xi8>) : tensor<2xi8>
  return
}

// -----

func.func @unsupported_element_type(
    %ctx: !hip.context, %input: tensor<2xbf16>, %shape: tensor<1xi64>) {
  %valid = arith.constant true
  %output = tensor.empty() : tensor<2xbf16>
  // expected-error @+1 {{unsupported input element type bf16}}
  %result = hip.expand(%ctx) valid(%valid)
    ins(%input, %shape : tensor<2xbf16>, tensor<1xi64>)
    outs(%output : tensor<2xbf16>) : tensor<2xbf16>
  return
}

// -----

func.func @invalid_shape_element_type(
    %ctx: !hip.context, %input: tensor<2xf32>, %shape: tensor<1xf32>) {
  %valid = arith.constant true
  %output = tensor.empty() : tensor<2xf32>
  // expected-error @+1 {{shape element type must be i32 or i64}}
  %result = hip.expand(%ctx) valid(%valid)
    ins(%input, %shape : tensor<2xf32>, tensor<1xf32>)
    outs(%output : tensor<2xf32>) : tensor<2xf32>
  return
}

// -----

func.func @dynamic_shape_length(
    %ctx: !hip.context, %input: tensor<2xf32>, %shape: tensor<?xi64>) {
  %valid = arith.constant true
  %output = tensor.empty() : tensor<2xf32>
  // expected-error @+1 {{shape length must be static}}
  %result = hip.expand(%ctx) valid(%valid)
    ins(%input, %shape : tensor<2xf32>, tensor<?xi64>)
    outs(%output : tensor<2xf32>) : tensor<2xf32>
  return
}

// -----

func.func @wrong_output_rank(
    %ctx: !hip.context, %input: tensor<2x3xf32>, %shape: tensor<1xi64>) {
  %valid = arith.constant true
  %output = tensor.empty() : tensor<3xf32>
  // expected-error @+1 {{output rank must equal max(input rank, shape length)}}
  %result = hip.expand(%ctx) valid(%valid)
    ins(%input, %shape : tensor<2x3xf32>, tensor<1xi64>)
    outs(%output : tensor<3xf32>) : tensor<3xf32>
  return
}

// -----

func.func @constant_target_mismatch(
    %ctx: !hip.context, %input: tensor<1xf32>) {
  %valid = arith.constant true
  %shape = arith.constant dense<[3]> : tensor<1xi64>
  %output = tensor.empty() : tensor<2xf32>
  // expected-error @+1 {{output extent contradicts constant target at axis 0}}
  %result = hip.expand(%ctx) valid(%valid)
    ins(%input, %shape : tensor<1xf32>, tensor<1xi64>)
    outs(%output : tensor<2xf32>) : tensor<2xf32>
  return
}

// -----

func.func @invalid_expected_extent(
    %ctx: !hip.context, %input: index, %target: index, %elements: index) {
  %valid = arith.constant true
  // expected-error @+1 {{expected_extent must be -1 or non-negative}}
  %next_valid, %extent, %next_elements = hip.checked_expand_extent(
      %ctx, %valid, %input, %target, %elements)
      expected_extent = -2 -> (i1, index, index)
  return
}
// -----

func.func @excessive_rank(
    %ctx: !hip.context,
    %input: tensor<1x1x1x1x1x1x1x1x1xf32>,
    %shape: tensor<9xi64>) {
  %valid = arith.constant true
  %output = tensor.empty() : tensor<1x1x1x1x1x1x1x1x1xf32>
  // expected-error @+1 {{output rank exceeds the runtime maximum of 8}}
  %result = hip.expand(%ctx) valid(%valid)
    ins(%input, %shape
      : tensor<1x1x1x1x1x1x1x1x1xf32>, tensor<9xi64>)
    outs(%output : tensor<1x1x1x1x1x1x1x1x1xf32>)
    : tensor<1x1x1x1x1x1x1x1x1xf32>
  return
}
