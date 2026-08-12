// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.transpose round-trip and verifier rules.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// A dynamic extent moves with its axis.
// CHECK-LABEL: func.func @transpose_swap(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<3x?xi64>,
// CHECK-SAME:    %[[INIT:.+]]: tensor<?x3xi64>) -> tensor<?x3xi64> {
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.transpose(%[[CTX]]) ins(%[[INPUT]] : tensor<3x?xi64>) outs(%[[INIT]] : tensor<?x3xi64>) {perm = array<i64: 1, 0>} : tensor<?x3xi64>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x3xi64>
func.func @transpose_swap(
    %ctx: !hipsr.context, %input: tensor<3x?xi64>,
    %init: tensor<?x3xi64>) -> tensor<?x3xi64> {
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<3x?xi64>)
      outs(%init : tensor<?x3xi64>) {perm = array<i64: 1, 0>} : tensor<?x3xi64>
  return %result : tensor<?x3xi64>
}

// -----

// A rank-3 rotation, where output axis i is input axis perm[i].
// CHECK-LABEL: func.func @transpose_rotate(
// CHECK:         hipsr.transpose(%{{.+}}) ins(%{{.+}} : tensor<2x3x4xf16>) outs(%{{.+}} : tensor<3x4x2xf16>) {perm = array<i64: 1, 2, 0>} : tensor<3x4x2xf16>
func.func @transpose_rotate(
    %ctx: !hipsr.context, %input: tensor<2x3x4xf16>,
    %init: tensor<3x4x2xf16>) -> tensor<3x4x2xf16> {
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<2x3x4xf16>)
      outs(%init : tensor<3x4x2xf16>)
      {perm = array<i64: 1, 2, 0>} : tensor<3x4x2xf16>
  return %result : tensor<3x4x2xf16>
}

// -----

// perm names one destination axis per input axis.
func.func @transpose_perm_length(
    %ctx: !hipsr.context, %input: tensor<2x3x4xf16>,
    %init: tensor<4x3x2xf16>) -> tensor<4x3x2xf16> {
  // expected-error@+1 {{perm length must equal the input rank; expected 3, got 2}}
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<2x3x4xf16>)
      outs(%init : tensor<4x3x2xf16>)
      {perm = array<i64: 1, 0>} : tensor<4x3x2xf16>
  return %result : tensor<4x3x2xf16>
}

// -----

// Repeating an axis would drop another one.
func.func @transpose_perm_repeated(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %init: tensor<2x2xf16>) -> tensor<2x2xf16> {
  // expected-error@+1 {{perm must be a permutation of [0, rank)}}
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<2x3xf16>)
      outs(%init : tensor<2x2xf16>) {perm = array<i64: 0, 0>} : tensor<2x2xf16>
  return %result : tensor<2x2xf16>
}

// -----

// An axis outside [0, rank) names nothing.
func.func @transpose_perm_out_of_range(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %init: tensor<3x2xf16>) -> tensor<3x2xf16> {
  // expected-error@+1 {{perm must be a permutation of [0, rank)}}
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<2x3xf16>)
      outs(%init : tensor<3x2xf16>) {perm = array<i64: 1, 2>} : tensor<3x2xf16>
  return %result : tensor<3x2xf16>
}

// -----

// Transpose moves elements; it does not convert them.
func.func @transpose_element_mismatch(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  // expected-error@+1 {{input and output element types must match}}
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<2x3xf16>)
      outs(%init : tensor<3x2xf32>) {perm = array<i64: 1, 0>} : tensor<3x2xf32>
  return %result : tensor<3x2xf32>
}

// -----

// Output axis i holds input axis perm[i], so the extents must line up.
func.func @transpose_output_shape(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{output shape must be the input shape permuted by perm}}
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<2x3xf16>)
      outs(%init : tensor<2x3xf16>) {perm = array<i64: 1, 0>} : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

// The permutation runs on the device.
func.func @transpose_host_input(
    %ctx: !hipsr.context,
    %input: memref<2x3xf16, #hipsr.mem<host>>,
    %init: memref<3x2xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{operand #1 must be ranked tensor or device memref}}
  hipsr.transpose(%ctx)
      ins(%input : memref<2x3xf16, #hipsr.mem<host>>)
      outs(%init : memref<3x2xf16, #hipsr.mem<device>>)
      {perm = array<i64: 1, 0>}
  return
}
