// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics

// Every slot is spelled out, including the ones ONNX lets a graph leave out:
// the conversion writes their defaults in.
func.func @missing_steps(
    %ctx: !hipsr.context, %data: tensor<8xf16, #hipsr.mem<device>>,
    %init: tensor<3xf16, #hipsr.mem<device>>)
    -> tensor<3xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{steps needs an operand or steps_attr}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      outs(%init : tensor<3xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 1>, ends_attr = array<i64: 4>,
       axes_attr = array<i64: 0>}
      : tensor<3xf16, #hipsr.mem<device>>
  return %result : tensor<3xf16, #hipsr.mem<device>>
}

// -----

// A slot holds its entries in one place, so the two forms are exclusive.
func.func @starts_twice(
    %ctx: !hipsr.context, %data: tensor<8xf16, #hipsr.mem<device>>,
    %starts: tensor<1xi64, #hipsr.mem<host>>,
    %init: tensor<3xf16, #hipsr.mem<device>>)
    -> tensor<3xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{starts cannot have both an operand and starts_attr}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      starts(%starts : tensor<1xi64, #hipsr.mem<host>>)
      outs(%init : tensor<3xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 1>, ends_attr = array<i64: 4>,
       axes_attr = array<i64: 0>, steps_attr = array<i64: 1>}
      : tensor<3xf16, #hipsr.mem<device>>
  return %result : tensor<3xf16, #hipsr.mem<device>>
}

// -----

// How many axes narrow is settled by the graph, so an operand carries the
// entries but never their count.
func.func @dynamic_window_length(
    %ctx: !hipsr.context, %data: tensor<8xf16, #hipsr.mem<device>>,
    %starts: tensor<?xi64, #hipsr.mem<host>>,
    %init: tensor<3xf16, #hipsr.mem<device>>)
    -> tensor<3xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{starts must hold a known number of entries}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      starts(%starts : tensor<?xi64, #hipsr.mem<host>>)
      outs(%init : tensor<3xf16, #hipsr.mem<device>>)
      {ends_attr = array<i64: 4>, axes_attr = array<i64: 0>,
       steps_attr = array<i64: 1>}
      : tensor<3xf16, #hipsr.mem<device>>
  return %result : tensor<3xf16, #hipsr.mem<device>>
}

// -----

// Each window entry describes one sliced axis, so every slot holds the same
// number of them.
func.func @window_length_mismatch(
    %ctx: !hipsr.context, %data: tensor<8xf16, #hipsr.mem<device>>,
    %init: tensor<3xf16, #hipsr.mem<device>>)
    -> tensor<3xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{starts, ends, axes and steps must hold one entry per sliced axis}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      outs(%init : tensor<3xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 1, 2>, ends_attr = array<i64: 4>}
      : tensor<3xf16, #hipsr.mem<device>>
  return %result : tensor<3xf16, #hipsr.mem<device>>
}

// -----

// A window operand is read on the host, and the op carries no copy from the
// device.
func.func @window_on_device(
    %ctx: !hipsr.context, %data: tensor<8xf16, #hipsr.mem<device>>,
    %starts: tensor<1xi64, #hipsr.mem<device>>,
    %init: tensor<3xf16, #hipsr.mem<device>>)
    -> tensor<3xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{operand #2 must be rank-1 i64 host tensor or memref, but got 'tensor<1xi64, #hipsr.mem<device>>'}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8xf16, #hipsr.mem<device>>)
      starts(%starts : tensor<1xi64, #hipsr.mem<device>>)
      outs(%init : tensor<3xf16, #hipsr.mem<device>>)
      {ends_attr = array<i64: 4>}
      : tensor<3xf16, #hipsr.mem<device>>
  return %result : tensor<3xf16, #hipsr.mem<device>>
}

// -----

// A window narrows axes; it never drops one, so the output keeps the data's
// rank. The verifier's axis-by-axis check counts on that.
func.func @output_rank_mismatch(
    %ctx: !hipsr.context, %data: tensor<8x4xf16, #hipsr.mem<device>>,
    %init: tensor<3xf16, #hipsr.mem<device>>)
    -> tensor<3xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{failed to verify that all of {data, init} have same rank}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8x4xf16, #hipsr.mem<device>>)
      outs(%init : tensor<3xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 1>, ends_attr = array<i64: 4>}
      : tensor<3xf16, #hipsr.mem<device>>
  return %result : tensor<3xf16, #hipsr.mem<device>>
}

// -----

// An output axis cannot come out longer than the data's, whether the window
// touches it or leaves it whole.
func.func @output_longer_than_data(
    %ctx: !hipsr.context, %data: tensor<8x4xf16, #hipsr.mem<device>>,
    %init: tensor<8x5xf16, #hipsr.mem<device>>)
    -> tensor<8x5xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{an output axis must not be longer than the data axis; axis 1 is 4 long in the data and 5 in the output}}
  %result = hipsr.slice(%ctx)
      ins(%data : tensor<8x4xf16, #hipsr.mem<device>>)
      outs(%init : tensor<8x5xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 1>, ends_attr = array<i64: 4>,
       axes_attr = array<i64: 1>, steps_attr = array<i64: 1>}
      : tensor<8x5xf16, #hipsr.mem<device>>
  return %result : tensor<8x5xf16, #hipsr.mem<device>>
}
