// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file --verify-diagnostics -hipsr-populate-shape-region -hipsr-populate-shape-region | FileCheck %s

// A populated placeholder is unchanged across repeated runs.
// CHECK-LABEL: func.func @already_populated(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<4x8xf32>) -> tensor<4x8xf16> {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<4x8xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16> shape_region {
// CHECK-NEXT: ^bb0(%[[SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<4x8xf32>) outs(%[[INIT]] : tensor<4x8xf16>) : tensor<4x8xf16>
// CHECK-NEXT: return %[[RESULT]] : tensor<4x8xf16>
// CHECK-NEXT: }
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
// CHECK-LABEL: func.func @whole_function_walk(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x8xf32>, %[[B:.+]]: tensor<8x16xf16>) -> tensor<?x16xf16> {
// CHECK-NEXT: %[[CAST_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16> shape_region {
// CHECK-NEXT: ^bb0(%[[INPUT_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: hipsr.shape_yield %[[INPUT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[CAST:.+]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32>) outs(%[[CAST_INIT]] : tensor<?x8xf16>) : tensor<?x8xf16>
// CHECK-NEXT: %[[MATMUL_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[CAST_INIT]], %[[B]] : tensor<?x8xf16>, tensor<8x16xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<?x16xf16> shape_region {
// CHECK-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// CHECK: hipsr.shape_yield %[[RESULT_SHAPE:.+]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[MATMUL:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[CAST]], %[[B]] : tensor<?x8xf16>, tensor<8x16xf16>) outs(%[[MATMUL_INIT]] : tensor<?x16xf16>) : tensor<?x16xf16>
// CHECK-NEXT: return %[[MATMUL]]
// CHECK-SAME: : tensor<?x16xf16>
// CHECK-NEXT: }
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

// Population drops stale inputs when canonicalization has not run and selects
// the final category from the consumer recipe.
// CHECK-LABEL: func.func @consumer_selects_category(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[LHS:.+]]: tensor<4x8xf32>, %[[STALE:.+]]: tensor<1xi64>, %[[RHS:.+]]: tensor<4x8xf32>) -> tensor<4x8xf32> {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<4x8xf32>, tensor<4x8xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32> shape_region {
// CHECK-NEXT: ^bb0(%[[LHS_SHAPE:.+]]: !shape.shape, %[[RHS_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: %[[SHAPE:.+]] = shape.broadcast %[[LHS_SHAPE]], %[[RHS_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.add(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<4x8xf32>, tensor<4x8xf32>) outs(%[[INIT]] : tensor<4x8xf32>) : tensor<4x8xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<4x8xf32>
// CHECK-NEXT: }
func.func @consumer_selects_category(
    %ctx: !hipsr.context, %lhs: tensor<4x8xf32>, %stale: tensor<1xi64>,
    %rhs: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %init = hipsr.placeholder(%ctx)
      ins(%lhs, %stale, %rhs
          : tensor<4x8xf32>, tensor<1xi64>, tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<barrier>} : tensor<4x8xf32>
  %result = hipsr.add(%ctx)
      ins(%lhs, %rhs : tensor<4x8xf32>, tensor<4x8xf32>)
      outs(%init : tensor<4x8xf32>) : tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}
