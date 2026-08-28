// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics
// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics -canonicalize | FileCheck %s

// A constant shape operand canonicalizes to shape_attr.
// CHECK-LABEL: func.func @expand_constant_shape(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[INPUT:.+]]: tensor<?x3xf16>,
// CHECK-SAME: %[[INIT:.+]]: tensor<?x?xf16>) -> tensor<?x?xf16> {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.expand(%[[CTX]]) ins(%[[INPUT]] : tensor<?x3xf16>) outs(%[[INIT]] : tensor<?x?xf16>) {shape_attr = array<i64: 4, 3>} : tensor<?x?xf16>
// CHECK-NEXT: return %[[RESULT]] : tensor<?x?xf16>
// CHECK-NEXT: }
func.func @expand_constant_shape(
    %ctx: !hipsr.context, %input: tensor<?x3xf16>,
    %init: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %shape = arith.constant dense<[4, 3]> : tensor<2xi64>
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
      outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
  return %result : tensor<?x?xf16>
}

// -----

func.func @expand_both_shapes(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %shape: tensor<2xi64>, %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{cannot have both shape operand and shape_attr attribute}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<2xi64>)
      outs(%init : tensor<2x3xf16>)
      {shape_attr = array<i64: 2, 3>} : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

func.func @expand_missing_shape(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{must have either shape operand or shape_attr attribute}}
  %result = hipsr.expand(%ctx)
      ins(%input : tensor<2x3xf16>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

// Shape must be a rank-1 extent vector.
func.func @expand_shape_rank(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %shape: tensor<1x2xi64>, %init: tensor<2x3xf16>)
    -> tensor<2x3xf16> {
  // expected-error@+1 {{shape must be rank-1}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<1x2xi64>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

// ONNX shape extents use i64 elements.
func.func @expand_shape_element_type(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %shape: tensor<2xi32>, %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{shape element type must be i64}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<2xi32>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

// The static shape length determines the output rank.
func.func @expand_dynamic_shape_length(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %shape: tensor<?xi64>, %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{shape length must be static}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<?xi64>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

// Expand preserves the input element type.
func.func @expand_element_mismatch(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %shape: tensor<2xi64>, %init: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // expected-error@+1 {{input and output element types must match}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<2xi64>)
      outs(%init : tensor<2x3xf32>) : tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// -----

// Output rank is max(input rank, requested shape length).
func.func @expand_output_rank(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %shape: tensor<4xi64>, %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{output rank must equal max(input rank, shape length); expected 4, got 2}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<4xi64>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

// Runtime shape values must be host-visible.
func.func @expand_device_shape(
    %ctx: !hipsr.context,
    %input: memref<2x3xf16, #hipsr.mem<device>>,
    %shape: memref<2xi64, #hipsr.mem<device>>,
    %init: memref<2x3xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{operand #2 must be ranked tensor or host memref}}
  hipsr.expand(%ctx)
      ins(%input, %shape : memref<2x3xf16, #hipsr.mem<device>>,
                            memref<2xi64, #hipsr.mem<device>>)
      outs(%init : memref<2x3xf16, #hipsr.mem<device>>)
  return
}
