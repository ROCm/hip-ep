// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.slice round-trip and verifier rules.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// A strided window over one axis; the other axis passes through whole.
// CHECK-LABEL: func.func @slice_strided_axis(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: tensor<8x4xf16>,
// CHECK-SAME:    %[[INIT:.+]]: tensor<3x4xf16>) -> tensor<3x4xf16> {
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.slice(%[[CTX]]) ins(%[[DATA]] : tensor<8x4xf16>) outs(%[[INIT]] : tensor<3x4xf16>) {axes = array<i64: 0>, ends = array<i64: 7>, starts = array<i64: 1>, steps = array<i64: 2>} : tensor<3x4xf16>
// CHECK-NEXT:    return %[[RESULT]] : tensor<3x4xf16>
func.func @slice_strided_axis(
    %ctx: !hipsr.context, %data: tensor<8x4xf16>,
    %init: tensor<3x4xf16>) -> tensor<3x4xf16> {
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8x4xf16>)
      outs(%init : tensor<3x4xf16>)
      {starts = array<i64: 1>, ends = array<i64: 7>, axes = array<i64: 0>,
       steps = array<i64: 2>} : tensor<3x4xf16>
  return %result : tensor<3x4xf16>
}

// -----

// Several axes at once, with an untouched axis free to stay dynamic.
// CHECK-LABEL: func.func @slice_multiple_axes(
// CHECK:         hipsr.slice(%{{.+}}) ins(%{{.+}} : tensor<?x8x4xf16>) outs(%{{.+}} : tensor<?x8x2xf16>) {axes = array<i64: 1, 2>, ends = array<i64: 8, 3>, starts = array<i64: 0, 1>, steps = array<i64: 1, 1>} : tensor<?x8x2xf16>
func.func @slice_multiple_axes(
    %ctx: !hipsr.context, %data: tensor<?x8x4xf16>,
    %init: tensor<?x8x2xf16>) -> tensor<?x8x2xf16> {
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<?x8x4xf16>)
      outs(%init : tensor<?x8x2xf16>)
      {starts = array<i64: 0, 1>, ends = array<i64: 8, 3>,
       axes = array<i64: 1, 2>, steps = array<i64: 1, 1>} : tensor<?x8x2xf16>
  return %result : tensor<?x8x2xf16>
}

// -----

// An empty window is in bounds and yields an empty axis.
// CHECK-LABEL: func.func @slice_empty_window(
// CHECK:         hipsr.slice(%{{.+}}) ins(%{{.+}} : tensor<8xf16>) outs(%{{.+}} : tensor<0xf16>) {axes = array<i64: 0>, ends = array<i64: 3>, starts = array<i64: 3>, steps = array<i64: 1>} : tensor<0xf16>
func.func @slice_empty_window(
    %ctx: !hipsr.context, %data: tensor<8xf16>,
    %init: tensor<0xf16>) -> tensor<0xf16> {
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16>)
      outs(%init : tensor<0xf16>)
      {starts = array<i64: 3>, ends = array<i64: 3>, axes = array<i64: 0>,
       steps = array<i64: 1>} : tensor<0xf16>
  return %result : tensor<0xf16>
}

// -----

// The four arrays describe the same axes, so their lengths agree.
func.func @slice_length_mismatch(
    %ctx: !hipsr.context, %data: tensor<8x4xf16>,
    %init: tensor<3x4xf16>) -> tensor<3x4xf16> {
  // expected-error@+1 {{starts, ends and steps must have one entry per axis; 1 axes, but 2 starts, 1 ends and 1 steps}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8x4xf16>)
      outs(%init : tensor<3x4xf16>)
      {starts = array<i64: 1, 0>, ends = array<i64: 7>, axes = array<i64: 0>,
       steps = array<i64: 2>} : tensor<3x4xf16>
  return %result : tensor<3x4xf16>
}

// -----

// ONNX normalizes a negative axis before it reaches the dialect.
func.func @slice_negative_axis(
    %ctx: !hipsr.context, %data: tensor<8x4xf16>,
    %init: tensor<8x2xf16>) -> tensor<8x2xf16> {
  // expected-error@+1 {{axes must be in [0, data rank); data rank is 2, got -1}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8x4xf16>)
      outs(%init : tensor<8x2xf16>)
      {starts = array<i64: 0>, ends = array<i64: 2>, axes = array<i64: -1>,
       steps = array<i64: 1>} : tensor<8x2xf16>
  return %result : tensor<8x2xf16>
}

// -----

// A second window on one axis would overwrite the first.
func.func @slice_repeated_axis(
    %ctx: !hipsr.context, %data: tensor<8x4xf16>,
    %init: tensor<2x4xf16>) -> tensor<2x4xf16> {
  // expected-error@+1 {{axes must be distinct; got axis 0 twice}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8x4xf16>)
      outs(%init : tensor<2x4xf16>)
      {starts = array<i64: 0, 0>, ends = array<i64: 4, 2>,
       axes = array<i64: 0, 0>, steps = array<i64: 1, 1>} : tensor<2x4xf16>
  return %result : tensor<2x4xf16>
}

// -----

// The bounds are already resolved against the extent, which a dynamic axis
// cannot supply.
func.func @slice_dynamic_axis(
    %ctx: !hipsr.context, %data: tensor<?xf16>,
    %init: tensor<2xf16>) -> tensor<2xf16> {
  // expected-error@+1 {{a sliced axis must be statically sized; axis 0 is dynamic}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<?xf16>)
      outs(%init : tensor<2xf16>)
      {starts = array<i64: 0>, ends = array<i64: 2>, axes = array<i64: 0>,
       steps = array<i64: 1>} : tensor<2xf16>
  return %result : tensor<2xf16>
}

// -----

// Reversing an axis is a separate operation.
func.func @slice_negative_step(
    %ctx: !hipsr.context, %data: tensor<8xf16>,
    %init: tensor<4xf16>) -> tensor<4xf16> {
  // expected-error@+1 {{steps must be positive; got -1 on axis 0}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16>)
      outs(%init : tensor<4xf16>)
      {starts = array<i64: 0>, ends = array<i64: 8>, axes = array<i64: 0>,
       steps = array<i64: -1>} : tensor<4xf16>
  return %result : tensor<4xf16>
}

// -----

// A zero step would not advance.
func.func @slice_zero_step(
    %ctx: !hipsr.context, %data: tensor<8xf16>,
    %init: tensor<8xf16>) -> tensor<8xf16> {
  // expected-error@+1 {{steps must be positive; got 0 on axis 0}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16>)
      outs(%init : tensor<8xf16>)
      {starts = array<i64: 0>, ends = array<i64: 8>, axes = array<i64: 0>,
       steps = array<i64: 0>} : tensor<8xf16>
  return %result : tensor<8xf16>
}

// -----

// ONNX saturates a bound past the end; the dialect takes it already resolved.
func.func @slice_end_past_extent(
    %ctx: !hipsr.context, %data: tensor<8xf16>,
    %init: tensor<8xf16>) -> tensor<8xf16> {
  // expected-error@+1 {{the window must lie within the axis; axis 0 has extent 8, got [0, 9)}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16>)
      outs(%init : tensor<8xf16>)
      {starts = array<i64: 0>, ends = array<i64: 9>, axes = array<i64: 0>,
       steps = array<i64: 1>} : tensor<8xf16>
  return %result : tensor<8xf16>
}

// -----

// A window that ends before it starts is empty in ONNX, not negative here.
func.func @slice_end_before_start(
    %ctx: !hipsr.context, %data: tensor<8xf16>,
    %init: tensor<0xf16>) -> tensor<0xf16> {
  // expected-error@+1 {{the window must lie within the axis; axis 0 has extent 8, got [4, 2)}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16>)
      outs(%init : tensor<0xf16>)
      {starts = array<i64: 4>, ends = array<i64: 2>, axes = array<i64: 0>,
       steps = array<i64: 1>} : tensor<0xf16>
  return %result : tensor<0xf16>
}

// -----

// Slice selects elements; it does not convert them.
func.func @slice_element_mismatch(
    %ctx: !hipsr.context, %data: tensor<8xf16>,
    %init: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{data and output element types must match}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16>)
      outs(%init : tensor<4xf32>)
      {starts = array<i64: 0>, ends = array<i64: 4>, axes = array<i64: 0>,
       steps = array<i64: 1>} : tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

// The stride divides the window, so half of eight at step two is two, not four.
func.func @slice_output_shape(
    %ctx: !hipsr.context, %data: tensor<8xf16>,
    %init: tensor<4xf16>) -> tensor<4xf16> {
  // expected-error@+1 {{output shape must be the data shape with each sliced axis narrowed to its window}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16>)
      outs(%init : tensor<4xf16>)
      {starts = array<i64: 0>, ends = array<i64: 4>, axes = array<i64: 0>,
       steps = array<i64: 2>} : tensor<4xf16>
  return %result : tensor<4xf16>
}

// -----

// An untouched axis keeps its extent.
func.func @slice_untouched_axis(
    %ctx: !hipsr.context, %data: tensor<8x4xf16>,
    %init: tensor<4x2xf16>) -> tensor<4x2xf16> {
  // expected-error@+1 {{output shape must be the data shape with each sliced axis narrowed to its window}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8x4xf16>)
      outs(%init : tensor<4x2xf16>)
      {starts = array<i64: 0>, ends = array<i64: 4>, axes = array<i64: 0>,
       steps = array<i64: 1>} : tensor<4x2xf16>
  return %result : tensor<4x2xf16>
}

// -----

// The window is taken on the device.
func.func @slice_host_data(
    %ctx: !hipsr.context,
    %data: memref<8xf16, #hipsr.mem<host>>,
    %init: memref<4xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{operand #1 must be ranked tensor or device memref}}
  hipsr.slice(%ctx)
      ins(%data : memref<8xf16, #hipsr.mem<host>>)
      outs(%init : memref<4xf16, #hipsr.mem<device>>)
      {starts = array<i64: 0>, ends = array<i64: 4>, axes = array<i64: 0>,
       steps = array<i64: 1>}
  return
}
