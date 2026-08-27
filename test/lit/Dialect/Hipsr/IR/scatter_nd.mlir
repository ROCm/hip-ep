// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.scatter_nd verifier rules. The operation round-trips in the conversion,
// shape-region, bufferization and lowering tests.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics

// Indices name positions, so they cannot be floating point.
func.func @float_indices(
    %ctx: !hipsr.context, %data: tensor<4x2xf16, #hipsr.mem<device>>,
    %ids: tensor<5x2xf32, #hipsr.mem<device>>,
    %updates: tensor<5xf16, #hipsr.mem<device>>,
    %init: tensor<4x2xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{indices element type must be an integer}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16, #hipsr.mem<device>>,
                                  tensor<5x2xf32, #hipsr.mem<device>>,
                                  tensor<5xf16, #hipsr.mem<device>>)
      outs(%init : tensor<4x2xf16, #hipsr.mem<device>>)
      : tensor<4x2xf16, #hipsr.mem<device>>
  return
}

// -----

// The updates are written into the data, so they share its element type.
func.func @updates_element_mismatch(
    %ctx: !hipsr.context, %data: tensor<4x2xf16, #hipsr.mem<device>>,
    %ids: tensor<5x2xi64, #hipsr.mem<device>>,
    %updates: tensor<5xf32, #hipsr.mem<device>>,
    %init: tensor<4x2xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{failed to verify that all of {data, updates, init} have same element type}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16, #hipsr.mem<device>>,
                                  tensor<5x2xi64, #hipsr.mem<device>>,
                                  tensor<5xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x2xf16, #hipsr.mem<device>>)
      : tensor<4x2xf16, #hipsr.mem<device>>
  return
}

// -----

// The output is a copy of the data with some slices overwritten.
func.func @output_shape(
    %ctx: !hipsr.context, %data: tensor<4x2xf16, #hipsr.mem<device>>,
    %ids: tensor<5x2xi64, #hipsr.mem<device>>,
    %updates: tensor<5xf16, #hipsr.mem<device>>,
    %init: tensor<5x2xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{output shape must match the data shape}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16, #hipsr.mem<device>>,
                                  tensor<5x2xi64, #hipsr.mem<device>>,
                                  tensor<5xf16, #hipsr.mem<device>>)
      outs(%init : tensor<5x2xf16, #hipsr.mem<device>>)
      : tensor<5x2xf16, #hipsr.mem<device>>
  return
}

// -----

// The trailing extent decides the updates shape, so it cannot wait for run
// time.
func.func @dynamic_index_depth(
    %ctx: !hipsr.context, %data: tensor<4x2xf16, #hipsr.mem<device>>,
    %ids: tensor<5x?xi64, #hipsr.mem<device>>,
    %updates: tensor<5xf16, #hipsr.mem<device>>,
    %init: tensor<4x2xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{indices must end in a static extent, which says how many data axes a row addresses}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16, #hipsr.mem<device>>,
                                  tensor<5x?xi64, #hipsr.mem<device>>,
                                  tensor<5xf16, #hipsr.mem<device>>)
      outs(%init : tensor<4x2xf16, #hipsr.mem<device>>)
      : tensor<4x2xf16, #hipsr.mem<device>>
  return
}

// -----

// A row addresses at least one axis.
func.func @empty_index_depth(
    %ctx: !hipsr.context, %data: tensor<4x2xf16, #hipsr.mem<device>>,
    %ids: tensor<5x0xi64, #hipsr.mem<device>>,
    %updates: tensor<5x4x2xf16, #hipsr.mem<device>>,
    %init: tensor<4x2xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{the trailing indices extent must be in [1, data rank]; data rank is 2, got 0}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16, #hipsr.mem<device>>,
                                  tensor<5x0xi64, #hipsr.mem<device>>,
                                  tensor<5x4x2xf16, #hipsr.mem<device>>)
      outs(%init : tensor<4x2xf16, #hipsr.mem<device>>)
      : tensor<4x2xf16, #hipsr.mem<device>>
  return
}

// -----

// And no more axes than the data has.
func.func @index_depth_too_deep(
    %ctx: !hipsr.context, %data: tensor<4x2xf16, #hipsr.mem<device>>,
    %ids: tensor<5x3xi64, #hipsr.mem<device>>,
    %updates: tensor<5xf16, #hipsr.mem<device>>,
    %init: tensor<4x2xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{the trailing indices extent must be in [1, data rank]; data rank is 2, got 3}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16, #hipsr.mem<device>>,
                                  tensor<5x3xi64, #hipsr.mem<device>>,
                                  tensor<5xf16, #hipsr.mem<device>>)
      outs(%init : tensor<4x2xf16, #hipsr.mem<device>>)
      : tensor<4x2xf16, #hipsr.mem<device>>
  return
}

// -----

// Addressing every axis leaves one element per row, so the updates lose the
// data extents.
func.func @updates_shape(
    %ctx: !hipsr.context, %data: tensor<4x2xf16, #hipsr.mem<device>>,
    %ids: tensor<5x2xi64, #hipsr.mem<device>>,
    %updates: tensor<5x2xf16, #hipsr.mem<device>>,
    %init: tensor<4x2xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{updates shape must be the leading indices extents followed by the data extents no row addresses}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16, #hipsr.mem<device>>,
                                  tensor<5x2xi64, #hipsr.mem<device>>,
                                  tensor<5x2xf16, #hipsr.mem<device>>)
      outs(%init : tensor<4x2xf16, #hipsr.mem<device>>)
      : tensor<4x2xf16, #hipsr.mem<device>>
  return
}
