// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -cse | FileCheck %s
// RUN: hip-mlir-opt %s -split-input-file -canonicalize | FileCheck %s

// Placeholder shape regions are IsolatedFromAbove. Equivalent outer and inner
// shape computations remain in their own scopes.
// CHECK-LABEL: func.func @cse_keeps_region_isolated
// CHECK: shape.num_elements
// CHECK: hipsr.placeholder
// CHECK: ^bb0(%[[INPUT_SHAPE:.+]]: !shape.shape):
// CHECK: shape.num_elements %[[INPUT_SHAPE]]
// CHECK: hipsr.shape_yield
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

// CHECK-LABEL: func.func @cse_keeps_region_constant
// CHECK: %[[OUTER:.+]] = arith.constant 0 : index
// CHECK: hipsr.placeholder
// CHECK: ^bb0(%{{.+}}: !shape.shape):
// CHECK: shape.{{(from_extents|const_shape)}}
// CHECK: hipsr.shape_yield
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
