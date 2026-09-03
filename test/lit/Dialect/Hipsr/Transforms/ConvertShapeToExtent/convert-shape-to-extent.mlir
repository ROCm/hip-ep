// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Positive coverage for -hipsr-convert-shape-to-extent.
//
// Each case checks its whole function with CHECK-NEXT, so the CHECK block is
// the expected output. Anything left over breaks the chain: a !shape.shape, an
// uninlined region, or an unrealized_conversion_cast.
//
// Error paths live in invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -hipsr-convert-shape-to-extent | FileCheck %s

// Every pattern at once, on operands whose rank the conversion can read.
//
// - The prologue inlines the shape.assuming and the scf.execute_region, so the
//   witness goes dead and the body flattens.
// - shape_of reads the rank off its operand, giving 2 and 3 extents.
// - broadcast left-pads with ones, so its result takes the longer rank, 3.
// - from_extents becomes tensor.from_elements, and concat becomes
//   tensor.concat with 2 + 3 extents. Upstream has no pattern for either.
// - const_size becomes arith.constant, and size_to_index folds into its
//   operand.
// - get_extent, num_elements, and preserve_shape are only retyped.
// CHECK-LABEL: func.func @convert_shape_ops(
// CHECK-SAME:      %[[A:.+]]: tensor<?x4xf16>, %[[B:.+]]: tensor<8x?x4xf16>) -> (tensor<?x4xf16, #hipsr.mem<device>>, index) {
// CHECK-NEXT:    %[[C4:.+]] = arith.constant 4 : index
// CHECK-NEXT:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[SA:.+]] = shape.shape_of %[[A]] : tensor<?x4xf16> -> tensor<2xindex>
// CHECK-NEXT:    %[[SB:.+]] = shape.shape_of %[[B]] : tensor<8x?x4xf16> -> tensor<3xindex>
// CHECK-NEXT:    %[[BCAST:.+]] = shape.broadcast %[[SA]], %[[SB]] : tensor<2xindex>, tensor<3xindex> -> tensor<3xindex>
// CHECK-NEXT:    %[[D0:.+]] = shape.get_extent %[[BCAST]], %[[C0]] : tensor<3xindex>, index -> index
// CHECK-NEXT:    %[[SA2:.+]] = shape.shape_of %[[A]] : tensor<?x4xf16> -> tensor<2xindex>
// CHECK-NEXT:    %[[CAT:.+]] = tensor.concat dim(0) %[[SA2]], %[[BCAST]] : (tensor<2xindex>, tensor<3xindex>) -> tensor<5xindex>
// CHECK-NEXT:    %[[COUNT:.+]] = shape.num_elements %[[CAT]] : tensor<5xindex> -> index
// CHECK-NEXT:    %[[SHAPE:.+]] = tensor.from_elements %[[D0]], %[[C4]] : tensor<2xindex>
// CHECK-NEXT:    %[[INIT:.+]] = tensor.empty(%[[D0]]) : tensor<?x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    hipsr.preserve_shape %[[SHAPE]], %[[INIT]] : tensor<2xindex>, tensor<?x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[INIT]], %[[COUNT]] : tensor<?x4xf16, #hipsr.mem<device>>, index
// CHECK-NEXT:  }
func.func @convert_shape_ops(%a: tensor<?x4xf16>, %b: tensor<8x?x4xf16>)
    -> (tensor<?x4xf16, #hipsr.mem<device>>, index) {
  %w = shape.const_witness true
  %bcast = shape.assuming %w -> !shape.shape {
    %r = scf.execute_region -> !shape.shape {
      %sa = shape.shape_of %a : tensor<?x4xf16> -> !shape.shape
      scf.yield %sa : !shape.shape
    }
    %sb = shape.shape_of %b : tensor<8x?x4xf16> -> !shape.shape
    %bc = shape.broadcast %r, %sb : !shape.shape, !shape.shape -> !shape.shape
    shape.assuming_yield %bc : !shape.shape
  }

  %c0 = shape.const_size 0
  %e0 = shape.get_extent %bcast, %c0 : !shape.shape, !shape.size -> !shape.size
  %d0 = shape.size_to_index %e0 : !shape.size

  %sa2 = shape.shape_of %a : tensor<?x4xf16> -> !shape.shape
  %cat = shape.concat %sa2, %bcast : !shape.shape, !shape.shape -> !shape.shape
  %n = shape.num_elements %cat : !shape.shape -> !shape.size
  %count = shape.size_to_index %n : !shape.size

  %c4 = arith.constant 4 : index
  %fe = shape.from_extents %d0, %c4 : index, index
  %init = tensor.empty(%d0) : tensor<?x4xf16, #hipsr.mem<device>>
  hipsr.preserve_shape %fe, %init : !shape.shape, tensor<?x4xf16, #hipsr.mem<device>>
  return %init, %count : tensor<?x4xf16, #hipsr.mem<device>>, index
}

// -----

// An unranked operand gives no rank to read, so the result falls back to
// tensor<?xindex> instead of failing. Upstream still lowers it; only the
// static indexing is lost.
// CHECK-LABEL: func.func @unranked_operand_falls_back(
// CHECK-SAME:      %[[A:.+]]: tensor<*xf16>) -> index {
// CHECK-NEXT:    %[[SHAPE:.+]] = shape.shape_of %[[A]] : tensor<*xf16> -> tensor<?xindex>
// CHECK-NEXT:    %[[COUNT:.+]] = shape.num_elements %[[SHAPE]] : tensor<?xindex> -> index
// CHECK-NEXT:    return %[[COUNT]] : index
// CHECK-NEXT:  }
func.func @unranked_operand_falls_back(%a: tensor<*xf16>) -> index {
  %s = shape.shape_of %a : tensor<*xf16> -> !shape.shape
  %n = shape.num_elements %s : !shape.shape -> !shape.size
  %i = shape.size_to_index %n : !shape.size
  return %i : index
}
