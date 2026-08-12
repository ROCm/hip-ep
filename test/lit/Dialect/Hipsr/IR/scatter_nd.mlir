// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.scatter_nd round-trip and verifier rules.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// Rows that address every data axis scatter single elements, so the updates
// are the leading indices extent alone.
// CHECK-LABEL: func.func @scatter_nd_elements(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: tensor<?x?x4096xf16>,
// CHECK-SAME:    %[[IDS:.+]]: tensor<?x3xi64>, %[[UPDATES:.+]]: tensor<?xf16>,
// CHECK-SAME:    %[[INIT:.+]]: tensor<?x?x4096xf16>) -> tensor<?x?x4096xf16> {
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.scatter_nd(%[[CTX]]) ins(%[[DATA]], %[[IDS]], %[[UPDATES]] : tensor<?x?x4096xf16>, tensor<?x3xi64>, tensor<?xf16>) outs(%[[INIT]] : tensor<?x?x4096xf16>) : tensor<?x?x4096xf16>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?x4096xf16>
func.func @scatter_nd_elements(
    %ctx: !hipsr.context, %data: tensor<?x?x4096xf16>, %ids: tensor<?x3xi64>,
    %updates: tensor<?xf16>,
    %init: tensor<?x?x4096xf16>) -> tensor<?x?x4096xf16> {
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<?x?x4096xf16>, tensor<?x3xi64>,
                                  tensor<?xf16>)
      outs(%init : tensor<?x?x4096xf16>) : tensor<?x?x4096xf16>
  return %result : tensor<?x?x4096xf16>
}

// -----

// A shallower row addresses whole slices, which the updates carry along.
// CHECK-LABEL: func.func @scatter_nd_slices(
// CHECK:         hipsr.scatter_nd(%{{.+}}) ins(%{{.+}}, %{{.+}}, %{{.+}} : tensor<4x8x2xf16>, tensor<5x1xi64>, tensor<5x8x2xf16>) outs(%{{.+}} : tensor<4x8x2xf16>) : tensor<4x8x2xf16>
func.func @scatter_nd_slices(
    %ctx: !hipsr.context, %data: tensor<4x8x2xf16>, %ids: tensor<5x1xi64>,
    %updates: tensor<5x8x2xf16>,
    %init: tensor<4x8x2xf16>) -> tensor<4x8x2xf16> {
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x8x2xf16>, tensor<5x1xi64>,
                                  tensor<5x8x2xf16>)
      outs(%init : tensor<4x8x2xf16>) : tensor<4x8x2xf16>
  return %result : tensor<4x8x2xf16>
}

// -----

// Indices name positions, so they cannot be floating point.
func.func @scatter_nd_float_indices(
    %ctx: !hipsr.context, %data: tensor<4x2xf16>, %ids: tensor<5x2xf32>,
    %updates: tensor<5xf16>, %init: tensor<4x2xf16>) -> tensor<4x2xf16> {
  // expected-error@+1 {{indices element type must be an integer}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16>, tensor<5x2xf32>,
                                  tensor<5xf16>)
      outs(%init : tensor<4x2xf16>) : tensor<4x2xf16>
  return %result : tensor<4x2xf16>
}

// -----

// The updates are written into the data, so they share its element type.
func.func @scatter_nd_updates_element_mismatch(
    %ctx: !hipsr.context, %data: tensor<4x2xf16>, %ids: tensor<5x2xi64>,
    %updates: tensor<5xf32>, %init: tensor<4x2xf16>) -> tensor<4x2xf16> {
  // expected-error@+1 {{data, updates and output element types must match}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16>, tensor<5x2xi64>,
                                  tensor<5xf32>)
      outs(%init : tensor<4x2xf16>) : tensor<4x2xf16>
  return %result : tensor<4x2xf16>
}

// -----

// The output is a copy of the data with some slices overwritten.
func.func @scatter_nd_output_shape(
    %ctx: !hipsr.context, %data: tensor<4x2xf16>, %ids: tensor<5x2xi64>,
    %updates: tensor<5xf16>, %init: tensor<5x2xf16>) -> tensor<5x2xf16> {
  // expected-error@+1 {{output shape must match the data shape}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16>, tensor<5x2xi64>,
                                  tensor<5xf16>)
      outs(%init : tensor<5x2xf16>) : tensor<5x2xf16>
  return %result : tensor<5x2xf16>
}

// -----

// A rank-0 indices tensor has no trailing extent to read.
func.func @scatter_nd_scalar_indices(
    %ctx: !hipsr.context, %data: tensor<4x2xf16>, %ids: tensor<i64>,
    %updates: tensor<2xf16>, %init: tensor<4x2xf16>) -> tensor<4x2xf16> {
  // expected-error@+1 {{indices must have rank at least one, to hold a trailing extent}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16>, tensor<i64>,
                                  tensor<2xf16>)
      outs(%init : tensor<4x2xf16>) : tensor<4x2xf16>
  return %result : tensor<4x2xf16>
}

// -----

// The updates shape follows from the trailing extent, so it must be known.
func.func @scatter_nd_dynamic_index_depth(
    %ctx: !hipsr.context, %data: tensor<4x2xf16>, %ids: tensor<5x?xi64>,
    %updates: tensor<5xf16>, %init: tensor<4x2xf16>) -> tensor<4x2xf16> {
  // expected-error@+1 {{the trailing indices extent selects which data axes a row addresses, so it must be static}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16>, tensor<5x?xi64>,
                                  tensor<5xf16>)
      outs(%init : tensor<4x2xf16>) : tensor<4x2xf16>
  return %result : tensor<4x2xf16>
}

// -----

// A row cannot address more axes than the data has.
func.func @scatter_nd_index_depth_too_deep(
    %ctx: !hipsr.context, %data: tensor<4x2xf16>, %ids: tensor<5x3xi64>,
    %updates: tensor<5xf16>, %init: tensor<4x2xf16>) -> tensor<4x2xf16> {
  // expected-error@+1 {{the trailing indices extent must be in [1, data rank]; data rank is 2, got 3}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16>, tensor<5x3xi64>,
                                  tensor<5xf16>)
      outs(%init : tensor<4x2xf16>) : tensor<4x2xf16>
  return %result : tensor<4x2xf16>
}

// -----

// A row addresses at least one axis.
func.func @scatter_nd_empty_index_depth(
    %ctx: !hipsr.context, %data: tensor<4x2xf16>, %ids: tensor<5x0xi64>,
    %updates: tensor<5x4x2xf16>, %init: tensor<4x2xf16>) -> tensor<4x2xf16> {
  // expected-error@+1 {{the trailing indices extent must be in [1, data rank]; data rank is 2, got 0}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16>, tensor<5x0xi64>,
                                  tensor<5x4x2xf16>)
      outs(%init : tensor<4x2xf16>) : tensor<4x2xf16>
  return %result : tensor<4x2xf16>
}

// -----

// Addressing every axis leaves an element per row, so the updates lose the
// data extents.
func.func @scatter_nd_updates_rank(
    %ctx: !hipsr.context, %data: tensor<4x2xf16>, %ids: tensor<5x2xi64>,
    %updates: tensor<5x2xf16>, %init: tensor<4x2xf16>) -> tensor<4x2xf16> {
  // expected-error@+1 {{updates shape must be the leading indices extents followed by the data extents no row addresses}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16>, tensor<5x2xi64>,
                                  tensor<5x2xf16>)
      outs(%init : tensor<4x2xf16>) : tensor<4x2xf16>
  return %result : tensor<4x2xf16>
}

// -----

// One update per row, so the leading extents line up.
func.func @scatter_nd_updates_extent(
    %ctx: !hipsr.context, %data: tensor<4x2xf16>, %ids: tensor<5x2xi64>,
    %updates: tensor<6xf16>, %init: tensor<4x2xf16>) -> tensor<4x2xf16> {
  // expected-error@+1 {{updates shape must be the leading indices extents followed by the data extents no row addresses}}
  %result = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16>, tensor<5x2xi64>,
                                  tensor<6xf16>)
      outs(%init : tensor<4x2xf16>) : tensor<4x2xf16>
  return %result : tensor<4x2xf16>
}

// -----

// The scatter runs on the device.
func.func @scatter_nd_host_updates(
    %ctx: !hipsr.context,
    %data: memref<4x2xf16, #hipsr.mem<device>>,
    %ids: memref<5x2xi64, #hipsr.mem<device>>,
    %updates: memref<5xf16, #hipsr.mem<host>>,
    %init: memref<4x2xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{operand #3 must be ranked tensor or device memref}}
  hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : memref<4x2xf16, #hipsr.mem<device>>,
                                  memref<5x2xi64, #hipsr.mem<device>>,
                                  memref<5xf16, #hipsr.mem<host>>)
      outs(%init : memref<4x2xf16, #hipsr.mem<device>>)
  return
}
