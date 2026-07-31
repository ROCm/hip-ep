// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// UNSUPPORTED: true
//
//===----------------------------------------------------------------------===//
// Shape regions are IsolatedFromAbove, so -cse and -canonicalize must not merge
// region-internal ops with identical ops in the enclosing function.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -cse | FileCheck %s
// RUN: hip-mlir-opt %s -split-input-file -canonicalize | FileCheck %s

// Function and region both compute shape.shape_of on the same value; CSE must
// not merge them (the region keeps its own over the entry-block arg).
// CHECK-LABEL: func.func @cse_keeps_region_isolated
// CHECK:       hipsr.cast
// CHECK:         ^bb0(%[[IN:.+]]: tensor<?x8xf32>):
// CHECK:         shape.shape_of %[[IN]]
// CHECK:         hipsr.shape_yield
func.func @cse_keeps_region_isolated(%ctx: !hipsr.context, %input: tensor<?x8xf32>,
                                     %init: tensor<?x8xf16>)
    -> (tensor<?x8xf16>, !shape.shape) {
  // Outer shape.shape_of over the operand -- must NOT be merged into the region.
  %outer = shape.shape_of %input : tensor<?x8xf32> -> !shape.shape
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
                  shape_region {
  ^bb0(%in: tensor<?x8xf32>):
    %shape = shape.shape_of %in : tensor<?x8xf32> -> tensor<2xindex>
    %c0 = arith.constant 0 : index
    %d0 = shape.get_extent %shape, %c0 : tensor<2xindex>, index -> index
    %c1 = arith.constant 1 : index
    %d1 = shape.get_extent %shape, %c1 : tensor<2xindex>, index -> index
    hipsr.shape_yield (%d0, %d1) : [f16]
  }
  return %0, %outer : tensor<?x8xf16>, !shape.shape
}

// -----

// An outer constant equals one inside the region; -cse / -canonicalize must not
// replace the region's constant with it (the region keeps its own).
// CHECK-LABEL: func.func @cse_keeps_region_constant
// CHECK:       hipsr.cast
// CHECK:         ^bb0(%[[IN:.+]]: tensor<?x8xf32>):
// CHECK:         arith.constant 0 : index
// CHECK:         hipsr.shape_yield
func.func @cse_keeps_region_constant(%ctx: !hipsr.context, %input: tensor<?x8xf32>,
                                     %init: tensor<?x8xf16>)
    -> (tensor<?x8xf16>, index) {
  // Outer constant -- must NOT replace the region's own constant.
  %outer_c0 = arith.constant 0 : index
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
                  shape_region {
  ^bb0(%in: tensor<?x8xf32>):
    %shape = shape.shape_of %in : tensor<?x8xf32> -> tensor<2xindex>
    %c0 = arith.constant 0 : index
    %d0 = shape.get_extent %shape, %c0 : tensor<2xindex>, index -> index
    %c1 = arith.constant 1 : index
    %d1 = shape.get_extent %shape, %c1 : tensor<2xindex>, index -> index
    hipsr.shape_yield (%d0, %d1) : [f16]
  }
  return %0, %outer_c0 : tensor<?x8xf16>, index
}
