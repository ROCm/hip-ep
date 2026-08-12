// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.gather round-trip and verifier rules.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// An embedding lookup: rank-2 indices replace the gathered axis, so the
// result outranks the table.
// CHECK-LABEL: func.func @gather_embedding(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context, %[[TABLE:.+]]: tensor<8x4xf16>,
// CHECK-SAME:    %[[IDS:.+]]: tensor<?x?xi64>,
// CHECK-SAME:    %[[INIT:.+]]: tensor<?x?x4xf16>) -> tensor<?x?x4xf16> {
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.gather(%[[CTX]]) ins(%[[TABLE]], %[[IDS]] : tensor<8x4xf16>, tensor<?x?xi64>) outs(%[[INIT]] : tensor<?x?x4xf16>) {axis = 0 : i64} : tensor<?x?x4xf16>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?x4xf16>
func.func @gather_embedding(
    %ctx: !hipsr.context, %table: tensor<8x4xf16>, %ids: tensor<?x?xi64>,
    %init: tensor<?x?x4xf16>) -> tensor<?x?x4xf16> {
  %result = hipsr.gather(%ctx)
      ins(%table, %ids : tensor<8x4xf16>, tensor<?x?xi64>)
      outs(%init : tensor<?x?x4xf16>) {axis = 0 : i64} : tensor<?x?x4xf16>
  return %result : tensor<?x?x4xf16>
}

// -----

// Rank-0 indices drop the gathered axis, which is how a graph reads one
// extent out of a shape vector.
// CHECK-LABEL: func.func @gather_scalar_index(
// CHECK:         hipsr.gather(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<2xi64>, tensor<i64>) outs(%{{.+}} : tensor<i64>) {axis = 0 : i64} : tensor<i64>
func.func @gather_scalar_index(
    %ctx: !hipsr.context, %shape: tensor<2xi64>, %index: tensor<i64>,
    %init: tensor<i64>) -> tensor<i64> {
  %result = hipsr.gather(%ctx)
      ins(%shape, %index : tensor<2xi64>, tensor<i64>)
      outs(%init : tensor<i64>) {axis = 0 : i64} : tensor<i64>
  return %result : tensor<i64>
}

// -----

// A trailing axis keeps the leading data extents in place.
// CHECK-LABEL: func.func @gather_inner_axis(
// CHECK:         hipsr.gather(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<2x3x4xf16>, tensor<5xi64>) outs(%{{.+}} : tensor<2x3x5xf16>) {axis = 2 : i64} : tensor<2x3x5xf16>
func.func @gather_inner_axis(
    %ctx: !hipsr.context, %data: tensor<2x3x4xf16>, %indices: tensor<5xi64>,
    %init: tensor<2x3x5xf16>) -> tensor<2x3x5xf16> {
  %result = hipsr.gather(%ctx)
      ins(%data, %indices : tensor<2x3x4xf16>, tensor<5xi64>)
      outs(%init : tensor<2x3x5xf16>) {axis = 2 : i64} : tensor<2x3x5xf16>
  return %result : tensor<2x3x5xf16>
}

// -----

// ONNX normalizes a negative axis before it reaches the dialect.
func.func @gather_negative_axis(
    %ctx: !hipsr.context, %data: tensor<2x3xf16>, %indices: tensor<5xi64>,
    %init: tensor<2x5xf16>) -> tensor<2x5xf16> {
  // expected-error@+1 {{axis must be in [0, data rank); data rank is 2, got -1}}
  %result = hipsr.gather(%ctx)
      ins(%data, %indices : tensor<2x3xf16>, tensor<5xi64>)
      outs(%init : tensor<2x5xf16>) {axis = -1 : i64} : tensor<2x5xf16>
  return %result : tensor<2x5xf16>
}

// -----

// Only the data's own axes can be gathered.
func.func @gather_axis_out_of_range(
    %ctx: !hipsr.context, %data: tensor<2x3xf16>, %indices: tensor<5xi64>,
    %init: tensor<2x5xf16>) -> tensor<2x5xf16> {
  // expected-error@+1 {{axis must be in [0, data rank); data rank is 2, got 2}}
  %result = hipsr.gather(%ctx)
      ins(%data, %indices : tensor<2x3xf16>, tensor<5xi64>)
      outs(%init : tensor<2x5xf16>) {axis = 2 : i64} : tensor<2x5xf16>
  return %result : tensor<2x5xf16>
}

// -----

// Indices name positions, so they cannot be floating point.
func.func @gather_float_indices(
    %ctx: !hipsr.context, %data: tensor<2x3xf16>, %indices: tensor<5xf32>,
    %init: tensor<5x3xf16>) -> tensor<5x3xf16> {
  // expected-error@+1 {{indices element type must be an integer}}
  %result = hipsr.gather(%ctx)
      ins(%data, %indices : tensor<2x3xf16>, tensor<5xf32>)
      outs(%init : tensor<5x3xf16>) {axis = 0 : i64} : tensor<5x3xf16>
  return %result : tensor<5x3xf16>
}

// -----

// Gather selects elements; it does not convert them.
func.func @gather_element_mismatch(
    %ctx: !hipsr.context, %data: tensor<2x3xf16>, %indices: tensor<5xi64>,
    %init: tensor<5x3xf32>) -> tensor<5x3xf32> {
  // expected-error@+1 {{data and output element types must match}}
  %result = hipsr.gather(%ctx)
      ins(%data, %indices : tensor<2x3xf16>, tensor<5xi64>)
      outs(%init : tensor<5x3xf32>) {axis = 0 : i64} : tensor<5x3xf32>
  return %result : tensor<5x3xf32>
}

// -----

// The gathered axis is replaced by the whole indices shape.
func.func @gather_output_shape(
    %ctx: !hipsr.context, %data: tensor<2x3xf16>, %indices: tensor<5xi64>,
    %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{output shape must be the data shape with axis replaced by the indices shape}}
  %result = hipsr.gather(%ctx)
      ins(%data, %indices : tensor<2x3xf16>, tensor<5xi64>)
      outs(%init : tensor<2x3xf16>) {axis = 0 : i64} : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

// The lookup runs on the device.
func.func @gather_host_data(
    %ctx: !hipsr.context,
    %data: memref<2x3xf16, #hipsr.mem<host>>,
    %indices: memref<5xi64, #hipsr.mem<device>>,
    %init: memref<5x3xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{operand #1 must be ranked tensor or device memref}}
  hipsr.gather(%ctx)
      ins(%data, %indices : memref<2x3xf16, #hipsr.mem<host>>,
                            memref<5xi64, #hipsr.mem<device>>)
      outs(%init : memref<5x3xf16, #hipsr.mem<device>>) {axis = 0 : i64}
  return
}
