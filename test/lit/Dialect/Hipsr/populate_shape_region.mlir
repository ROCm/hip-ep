// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file --verify-diagnostics -hipsr-populate-shape-region -hipsr-populate-shape-region | FileCheck %s

// A populated placeholder is unchanged across repeated runs.
// CHECK-LABEL: func.func @already_populated
// CHECK: %[[INIT:.+]] = hipsr.placeholder
// CHECK-SAME: shape_region {
// CHECK-NEXT: ^bb0(%[[SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NOT: shape.broadcast
func.func @already_populated(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%input_shape: !shape.shape):
    hipsr.shape_yield %input_shape : !shape.shape
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Every empty placeholder in a shape-graph chain is populated in one run.
// CHECK-LABEL: func.func @whole_function_walk
// CHECK: %[[CAST_INIT:.+]] = hipsr.placeholder
// CHECK-SAME: shape_region {
// CHECK: hipsr.shape_yield
// CHECK: %[[CAST:.+]] = hipsr.cast
// CHECK-NOT: shape_region
// CHECK: %[[MATMUL_INIT:.+]] = hipsr.placeholder
// CHECK-SAME: ins(%[[CAST_INIT]]
// CHECK-SAME: shape_region {
// CHECK: shape.cstr_eq
// CHECK: hipsr.shape_yield
// CHECK: %[[MATMUL:.+]] = hipsr.matmul
// CHECK-NOT: shape_region
// CHECK-NEXT: return %[[MATMUL]]
func.func @whole_function_walk(
    %ctx: !hipsr.context, %input: tensor<?x8xf32>,
    %b: tensor<8x16xf16>) -> tensor<?x16xf16> {
  %cast_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %cast = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
      outs(%cast_init : tensor<?x8xf16>) : tensor<?x8xf16>
  %matmul_init = hipsr.placeholder(%ctx)
      ins(%cast_init, %b : tensor<?x8xf16>, tensor<8x16xf16>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x16xf16>
  %matmul = hipsr.matmul(%ctx)
      ins(%cast, %b : tensor<?x8xf16>, tensor<8x16xf16>)
      outs(%matmul_init : tensor<?x16xf16>) : tensor<?x16xf16>
  return %matmul : tensor<?x16xf16>
}

// -----

// The consumer recipe selects the final category for an empty placeholder.
// CHECK-LABEL: func.func @consumer_selects_category
// CHECK: %[[INIT:.+]] = hipsr.placeholder
// CHECK-SAME: #hipsr.placeholder_type<normal>
// CHECK-SAME: shape_region {
// CHECK-NEXT: ^bb0(%[[SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: hipsr.shape_yield %[[SHAPE]] : !shape.shape
func.func @consumer_selects_category(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<barrier>} : tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Functions without placeholders are unchanged.
// CHECK-LABEL: func.func @no_placeholder
// CHECK-NEXT: return
func.func @no_placeholder() {
  return
}
