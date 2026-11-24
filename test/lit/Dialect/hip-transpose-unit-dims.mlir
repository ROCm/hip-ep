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
// Negative cases: each guard of the pattern has one -- a transpose that
// reorders a non-unit dimension, is in memref form, declares a result type that
// disagrees with the permutation, or has no reassociation must stay.

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

  // Positive: several unit dims, not adjacent, move at once. The non-unit dims
  // (4 and 8) keep their relative order.
  // CHECK-LABEL: func.func @multiple_unit_dims
  // CHECK-NOT:     hip.transpose
  // CHECK:         %[[C:.*]] = tensor.collapse_shape %arg1 {{\[\[0, 1\], \[2, 3\]\]}} : tensor<1x4x1x8xf32> into tensor<4x8xf32>
  // CHECK:         %[[E:.*]] = tensor.expand_shape %[[C]] {{\[\[0\], \[1, 2, 3\]\]}} output_shape [4, 1, 8, 1] : tensor<4x8xf32> into tensor<4x1x8x1xf32>
  // CHECK:         return %[[E]]
  func.func @multiple_unit_dims(
      %ctx: !hip.context,
      %x: tensor<1x4x1x8xf32>) -> tensor<4x1x8x1xf32> {
    %d = tensor.empty() : tensor<4x1x8x1xf32>
    %0 = hip.transpose(%ctx) ins(%x : tensor<1x4x1x8xf32>)
                             outs(%d : tensor<4x1x8x1xf32>)
                             {perm = [1, 2, 3, 0]} : tensor<4x1x8x1xf32>
    return %0 : tensor<4x1x8x1xf32>
  }

  // Negative: memref form. The pattern only applies to tensors, so this
  // transpose is left for the later passes.
  // CHECK-LABEL: func.func @memref_form
  // CHECK:         hip.transpose(%arg0) ins(%arg1 : memref<1x169x768xf16>) outs(%arg2 : memref<169x1x768xf16>)
  // CHECK-NOT:     memref.collapse_shape
  func.func @memref_form(
      %ctx: !hip.context,
      %x: memref<1x169x768xf16>, %d: memref<169x1x768xf16>) {
    hip.transpose(%ctx) ins(%x : memref<1x169x768xf16>)
                        outs(%d : memref<169x1x768xf16>)
                        {perm = [1, 0, 2]}
    return
  }

  // Negative: the declared result type (5x1x8) is not the input type permuted
  // by `perm` (4x1x8). The verifier does not relate the two, so the pattern
  // must reject it rather than build a reshape from the wrong type.
  // CHECK-LABEL: func.func @result_type_disagrees
  // CHECK:         hip.transpose
  // CHECK-NOT:     tensor.collapse_shape
  func.func @result_type_disagrees(
      %ctx: !hip.context,
      %x: tensor<1x4x8xf32>) -> tensor<5x1x8xf32> {
    %d = tensor.empty() : tensor<5x1x8xf32>
    %0 = hip.transpose(%ctx) ins(%x : tensor<1x4x8xf32>)
                             outs(%d : tensor<5x1x8xf32>)
                             {perm = [1, 0, 2]} : tensor<5x1x8xf32>
    return %0 : tensor<5x1x8xf32>
  }

  // Negative (conservative): a static input dim whose result dim is dynamic.
  // The types are compatible, but no reassociation exists between the static
  // collapsed extent and the dynamic result extent, so the transpose stays.
  // CHECK-LABEL: func.func @static_input_dynamic_result
  // CHECK:         hip.transpose
  // CHECK-NOT:     tensor.collapse_shape
  func.func @static_input_dynamic_result(
      %ctx: !hip.context,
      %x: tensor<1x4x8xf32>, %n: index) -> tensor<?x1x8xf32> {
    %d = tensor.empty(%n) : tensor<?x1x8xf32>
    %0 = hip.transpose(%ctx) ins(%x : tensor<1x4x8xf32>)
                             outs(%d : tensor<?x1x8xf32>)
                             {perm = [1, 0, 2]} : tensor<?x1x8xf32>
    return %0 : tensor<?x1x8xf32>
  }
}
