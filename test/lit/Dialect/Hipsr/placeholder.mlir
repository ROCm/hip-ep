// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file -cse %s | FileCheck %s --check-prefix=CSE

// Placeholders preserve dynamic, static, and rank-0 result types. The dynamic
// case carries one shape-region block; the others have zero blocks.
// CHECK-LABEL: func.func @tensor_forms(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME: %[[DYNAMIC_INPUT:.*]]: tensor<?x?xf32>,
// CHECK-SAME: %[[STATIC_INPUT:.*]]: tensor<4x8xf32>,
// CHECK-SAME: %[[SCALAR_INPUT:.*]]: tensor<f32>)
// CHECK-SAME: -> (tensor<?x?xf16>, tensor<4x8xi64>, tensor<f16>) {
// CHECK-NEXT: %[[DYNAMIC_INIT:.*]] = hipsr.placeholder(%[[CTX]], %[[DYNAMIC_INPUT]] : tensor<?x?xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<?x?xf16> shape_region {
// CHECK-NEXT: ^bb0(%{{.*}}: !hipsr.context, %[[SHAPE_INPUT:.*]]: tensor<?x?xf32>):
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[D0:.*]] = tensor.dim %[[SHAPE_INPUT]], %[[C0]] : tensor<?x?xf32>
// CHECK-NEXT: %[[C8:.*]] = arith.constant 8 : index
// CHECK-NEXT: hipsr.shape_yield (%[[D0]], %[[C8]]) : [f16]
// CHECK-NEXT: }
// CHECK-NEXT: %[[DYNAMIC_RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[DYNAMIC_INPUT]] : tensor<?x?xf32>) outs(%[[DYNAMIC_INIT]] : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK-NEXT: %[[STATIC_INIT:.*]] = hipsr.placeholder(%[[CTX]], %[[STATIC_INPUT]] : tensor<4x8xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<4x8xi64>
// CHECK-NEXT: %[[STATIC_RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[STATIC_INPUT]] : tensor<4x8xf32>) outs(%[[STATIC_INIT]] : tensor<4x8xi64>) : tensor<4x8xi64>
// CHECK-NEXT: %[[SCALAR_INIT:.*]] = hipsr.placeholder(%[[CTX]], %[[SCALAR_INPUT]] : tensor<f32>) {type = #hipsr.placeholder_type<normal>} : tensor<f16>
// CHECK-NEXT: %[[SCALAR_RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[SCALAR_INPUT]] : tensor<f32>) outs(%[[SCALAR_INIT]] : tensor<f16>) : tensor<f16>
// CHECK-NEXT: return %[[DYNAMIC_RESULT]], %[[STATIC_RESULT]], %[[SCALAR_RESULT]] : tensor<?x?xf16>, tensor<4x8xi64>, tensor<f16>
// CHECK-NEXT: }
func.func @tensor_forms(
    %ctx: !hipsr.context, %dynamic_input: tensor<?x?xf32>,
    %static_input: tensor<4x8xf32>, %scalar_input: tensor<f32>)
    -> (tensor<?x?xf16>, tensor<4x8xi64>, tensor<f16>) {
  %dynamic_init = hipsr.placeholder(
      %ctx, %dynamic_input : tensor<?x?xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x?xf16>
      shape_region {
  ^bb0(%shape_ctx: !hipsr.context, %shape_input: tensor<?x?xf32>):
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %shape_input, %c0 : tensor<?x?xf32>
    %c8 = arith.constant 8 : index
    hipsr.shape_yield (%d0, %c8) : [f16]
  }
  %dynamic_result = hipsr.cast(%ctx) ins(%dynamic_input : tensor<?x?xf32>)
      outs(%dynamic_init : tensor<?x?xf16>) : tensor<?x?xf16>
  %static_init = hipsr.placeholder(
      %ctx, %static_input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xi64>
  %static_result = hipsr.cast(%ctx) ins(%static_input : tensor<4x8xf32>)
      outs(%static_init : tensor<4x8xi64>) : tensor<4x8xi64>
  %scalar_init = hipsr.placeholder(
      %ctx, %scalar_input : tensor<f32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<f16>
  %scalar_result = hipsr.cast(%ctx) ins(%scalar_input : tensor<f32>)
      outs(%scalar_init : tensor<f16>) : tensor<f16>
  return %dynamic_result, %static_result, %scalar_result
      : tensor<?x?xf16>, tensor<4x8xi64>, tensor<f16>
}

// -----

// CSE must not merge otherwise identical placeholders tied to different ops.
// CSE-LABEL: func.func @cse_keeps_placeholders(
// CSE-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4x8xf32>)
// CSE-SAME: -> (tensor<4x8xf16>, tensor<4x8xf16>) {
// CSE-NEXT: %[[LHS_INIT:.*]] = hipsr.placeholder(%[[CTX]], %[[INPUT]] : tensor<4x8xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
// CSE-NEXT: %[[RHS_INIT:.*]] = hipsr.placeholder(%[[CTX]], %[[INPUT]] : tensor<4x8xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
// CSE-NEXT: %[[LHS_RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<4x8xf32>) outs(%[[LHS_INIT]] : tensor<4x8xf16>) : tensor<4x8xf16>
// CSE-NEXT: %[[RHS_RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<4x8xf32>) outs(%[[RHS_INIT]] : tensor<4x8xf16>) : tensor<4x8xf16>
// CSE-NEXT: return %[[LHS_RESULT]], %[[RHS_RESULT]] : tensor<4x8xf16>, tensor<4x8xf16>
// CSE-NEXT: }
func.func @cse_keeps_placeholders(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>)
    -> (tensor<4x8xf16>, tensor<4x8xf16>) {
  %lhs_init = hipsr.placeholder(
      %ctx, %input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %rhs_init = hipsr.placeholder(
      %ctx, %input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %lhs_result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%lhs_init : tensor<4x8xf16>) : tensor<4x8xf16>
  %rhs_result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%rhs_init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %lhs_result, %rhs_result : tensor<4x8xf16>, tensor<4x8xf16>
}
