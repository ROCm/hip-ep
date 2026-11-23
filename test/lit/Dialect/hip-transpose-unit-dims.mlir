// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --canonicalize %s | FileCheck %s

// What this file tests
// --------------------
// `UnitDimTransposeToReshape` -- a `hip.transpose` that only moves dimensions
// of extent 1 keeps the memory layout, so it becomes a collapse/expand pair
// (views) instead of a copy. See
// lib/Dialect/IR/HipTransposeUnitDimCanonicalize.cpp.
//
// Negative cases: a transpose that reorders a non-unit dimension must stay.

module {
  // Positive: the unit dim moves across a non-unit one.
  // CHECK-LABEL: func.func @unit_dim_moves
  // CHECK-NOT:     hip.transpose
  // CHECK:         %[[C:.*]] = tensor.collapse_shape %arg1 {{\[\[0, 1\], \[2\]\]}} : tensor<1x169x768xf16> into tensor<169x768xf16>
  // CHECK:         %[[E:.*]] = tensor.expand_shape %[[C]] {{\[\[0\], \[1, 2\]\]}} output_shape [169, 1, 768] : tensor<169x768xf16> into tensor<169x1x768xf16>
  // CHECK:         return %[[E]]
  func.func @unit_dim_moves(
      %ctx: !hip.context,
      %x: tensor<1x169x768xf16>) -> tensor<169x1x768xf16> {
    %d = tensor.empty() : tensor<169x1x768xf16>
    %0 = hip.transpose(%ctx) ins(%x : tensor<1x169x768xf16>)
                             outs(%d : tensor<169x1x768xf16>)
                             {perm = [1, 0, 2]} : tensor<169x1x768xf16>
    return %0 : tensor<169x1x768xf16>
  }

  // Positive: a dynamic non-unit dim keeps its place; its extent is read from
  // the input.
  // CHECK-LABEL: func.func @dynamic_dim
  // CHECK-NOT:     hip.transpose
  // CHECK:         %[[D:.*]] = tensor.dim %arg1, %c1
  // CHECK:         %[[C:.*]] = tensor.collapse_shape %arg1 {{\[\[0, 1\], \[2\]\]}} : tensor<1x?x8xf32> into tensor<?x8xf32>
  // CHECK:         tensor.expand_shape %[[C]] {{\[\[0\], \[1, 2\]\]}} output_shape [%[[D]], 1, 8] : tensor<?x8xf32> into tensor<?x1x8xf32>
  func.func @dynamic_dim(
      %ctx: !hip.context,
      %x: tensor<1x?x8xf32>, %n: index) -> tensor<?x1x8xf32> {
    %d = tensor.empty(%n) : tensor<?x1x8xf32>
    %0 = hip.transpose(%ctx) ins(%x : tensor<1x?x8xf32>)
                             outs(%d : tensor<?x1x8xf32>)
                             {perm = [1, 0, 2]} : tensor<?x1x8xf32>
    return %0 : tensor<?x1x8xf32>
  }

  // Positive: two unit dims swap, so input and result types are equal and the
  // transpose is replaced by its input.
  // CHECK-LABEL: func.func @identity_types
  // CHECK-NOT:     hip.transpose
  // CHECK-NOT:     tensor.collapse_shape
  // CHECK:         return %arg1
  func.func @identity_types(
      %ctx: !hip.context,
      %x: tensor<1x1x16xf32>) -> tensor<1x1x16xf32> {
    %d = tensor.empty() : tensor<1x1x16xf32>
    %0 = hip.transpose(%ctx) ins(%x : tensor<1x1x16xf32>)
                             outs(%d : tensor<1x1x16xf32>)
                             {perm = [1, 0, 2]} : tensor<1x1x16xf32>
    return %0 : tensor<1x1x16xf32>
  }

  // Negative: a non-unit dim moves.
  // CHECK-LABEL: func.func @non_unit_dim_moves
  // CHECK:         hip.transpose
  // CHECK-NOT:     tensor.collapse_shape
  func.func @non_unit_dim_moves(
      %ctx: !hip.context,
      %x: tensor<1x4x8xf32>) -> tensor<1x8x4xf32> {
    %d = tensor.empty() : tensor<1x8x4xf32>
    %0 = hip.transpose(%ctx) ins(%x : tensor<1x4x8xf32>)
                             outs(%d : tensor<1x8x4xf32>)
                             {perm = [0, 2, 1]} : tensor<1x8x4xf32>
    return %0 : tensor<1x8x4xf32>
  }

  // Negative: two dynamic dims swap. Either could be 1 at run time, but they
  // are not statically unit, so the transpose stays.
  // CHECK-LABEL: func.func @dynamic_dims_swap
  // CHECK:         hip.transpose
  // CHECK-NOT:     tensor.collapse_shape
  func.func @dynamic_dims_swap(
      %ctx: !hip.context,
      %x: tensor<?x?xf32>, %a: index, %b: index) -> tensor<?x?xf32> {
    %d = tensor.empty(%b, %a) : tensor<?x?xf32>
    %0 = hip.transpose(%ctx) ins(%x : tensor<?x?xf32>)
                             outs(%d : tensor<?x?xf32>)
                             {perm = [1, 0]} : tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
}
