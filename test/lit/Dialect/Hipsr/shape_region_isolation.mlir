// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -cse | FileCheck %s --check-prefixes=COMMON,CSE
// RUN: hip-mlir-opt %s -split-input-file -canonicalize | FileCheck %s --check-prefixes=COMMON,CANONICALIZE

// Placeholder shape regions are IsolatedFromAbove. Equivalent outer and inner
// shape computations remain in their own scopes.
// COMMON-LABEL: func.func @cse_keeps_region_isolated(
// COMMON-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x8xf32>) -> (tensor<?x8xf16>, !shape.size) {
// COMMON-NEXT: %[[OUTER_SHAPE:.+]] = shape.shape_of %[[INPUT]] : tensor<?x8xf32> -> !shape.shape
// COMMON-NEXT: %[[OUTER_ELEMENTS:.+]] = shape.num_elements %[[OUTER_SHAPE]] : !shape.shape -> !shape.size
// COMMON-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16> shape_region {
// COMMON-NEXT: ^bb0(%[[INPUT_SHAPE:.+]]: !shape.shape):
// COMMON-NEXT: %[[ELEMENTS:.+]] = shape.num_elements %[[INPUT_SHAPE]] : !shape.shape -> !shape.size
// COMMON-NEXT: %[[EXTENT:.+]] = shape.size_to_index %[[ELEMENTS]] : !shape.size
// COMMON-NEXT: %[[RESULT_SHAPE:.+]] = shape.from_extents %[[EXTENT]] : index
// COMMON-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// COMMON-NEXT: }
// COMMON-NEXT: %[[RESULT:.+]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32>) outs(%[[INIT]] : tensor<?x8xf16>) : tensor<?x8xf16>
// COMMON-NEXT: return %[[RESULT]], %[[OUTER_ELEMENTS]] : tensor<?x8xf16>, !shape.size
// COMMON-NEXT: }
func.func @cse_keeps_region_isolated(
    %ctx: !hipsr.context, %input: tensor<?x8xf32>)
    -> (tensor<?x8xf16>, !shape.size) {
  %outer_shape = shape.shape_of %input : tensor<?x8xf32> -> !shape.shape
  %outer_elements = shape.num_elements %outer_shape
      : !shape.shape -> !shape.size
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
      shape_region {
  ^bb0(%input_shape: !shape.shape):
    %elements = shape.num_elements %input_shape
        : !shape.shape -> !shape.size
    %extent = shape.size_to_index %elements : !shape.size
    %result_shape = shape.from_extents %extent : index
    hipsr.shape_yield %result_shape : !shape.shape
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %result, %outer_elements : tensor<?x8xf16>, !shape.size
}

// -----

// COMMON-LABEL: func.func @cse_keeps_region_constant(
// COMMON-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x8xf32>) -> (tensor<?x8xf16>, index) {
// COMMON-NEXT: %[[OUTER:.+]] = arith.constant 0 : index
// COMMON-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16> shape_region {
// COMMON-NEXT: ^bb0(%[[INPUT_SHAPE:.+]]: !shape.shape):
// CSE-NEXT: %[[INNER:.+]] = arith.constant 0 : index
// CSE-NEXT: %[[RESULT_SHAPE:.+]] = shape.from_extents %[[INNER]] : index
// CANONICALIZE-NEXT: %[[RESULT_SHAPE:.+]] = shape.const_shape [0] : !shape.shape
// COMMON-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// COMMON-NEXT: }
// COMMON-NEXT: %[[RESULT:.+]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32>) outs(%[[INIT]] : tensor<?x8xf16>) : tensor<?x8xf16>
// COMMON-NEXT: return %[[RESULT]], %[[OUTER]] : tensor<?x8xf16>, index
// COMMON-NEXT: }
func.func @cse_keeps_region_constant(
    %ctx: !hipsr.context, %input: tensor<?x8xf32>)
    -> (tensor<?x8xf16>, index) {
  %outer_c0 = arith.constant 0 : index
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
      shape_region {
  ^bb0(%input_shape: !shape.shape):
    %c0 = arith.constant 0 : index
    %result_shape = shape.from_extents %c0 : index
    hipsr.shape_yield %result_shape : !shape.shape
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %result, %outer_c0 : tensor<?x8xf16>, index
}
