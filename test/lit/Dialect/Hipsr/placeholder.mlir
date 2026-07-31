// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file -cse %s | FileCheck %s --check-prefix=CSE
// RUN: hip-mlir-opt --split-input-file -canonicalize %s | FileCheck %s --check-prefix=CANONICALIZE

// Normal regions omit context and receive one !shape.shape per input.
// A scalar result still yields one shape value with no extents.
// CHECK-LABEL: func.func @normal_forms(
// CHECK: %[[DYNAMIC_INIT:.+]] = hipsr.placeholder
// CHECK-SAME: #hipsr.placeholder_type<normal>
// CHECK-SAME: shape_region {
// CHECK-NEXT: ^bb0(%[[DYNAMIC_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: hipsr.shape_yield %[[DYNAMIC_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK: %[[SCALAR_INIT:.+]] = hipsr.placeholder
// CHECK-SAME: tensor<f16> shape_region {
// CHECK-NEXT: ^bb0(%{{.+}}: !shape.shape):
// CHECK-NEXT: %[[SCALAR_SHAPE:.+]] = shape.from_extents
// CHECK-NEXT: hipsr.shape_yield %[[SCALAR_SHAPE]] : !shape.shape
func.func @normal_forms(
    %ctx: !hipsr.context, %dynamic_input: tensor<?x?xf32>,
    %scalar_input: tensor<f32>) -> (tensor<?x?xf16>, tensor<f16>) {
  %dynamic_init = hipsr.placeholder(%ctx)
      ins(%dynamic_input : tensor<?x?xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x?xf16>
      shape_region {
  ^bb0(%input_shape: !shape.shape):
    hipsr.shape_yield %input_shape : !shape.shape
  }
  %dynamic_result = hipsr.cast(%ctx)
      ins(%dynamic_input : tensor<?x?xf32>)
      outs(%dynamic_init : tensor<?x?xf16>) : tensor<?x?xf16>

  %scalar_init = hipsr.placeholder(%ctx)
      ins(%scalar_input : tensor<f32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<f16>
      shape_region {
  ^bb0(%input_shape: !shape.shape):
    %scalar_shape = "shape.from_extents"() : () -> !shape.shape
    hipsr.shape_yield %scalar_shape : !shape.shape
  }
  %scalar_result = hipsr.cast(%ctx)
      ins(%scalar_input : tensor<f32>)
      outs(%scalar_init : tensor<f16>) : tensor<f16>
  return %dynamic_result, %scalar_result : tensor<?x?xf16>, tensor<f16>
}

// -----

// Barrier regions preserve the context and tensor inputs.
// CHECK-LABEL: func.func @barrier_layout(
// CHECK: %[[INIT:.+]] = hipsr.placeholder
// CHECK-SAME: #hipsr.placeholder_type<barrier>
// CHECK-SAME: shape_region {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[INPUT:.+]]: tensor<?x3xf16>, %{{.+}}: tensor<2xi64>):
// CHECK-NEXT: %[[SHAPE:.+]] = shape.shape_of %[[INPUT]]
// CHECK-NEXT: hipsr.shape_yield %[[SHAPE]] : !shape.shape
func.func @barrier_layout(
    %ctx: !hipsr.context, %input: tensor<?x3xf16>, %shape: tensor<2xi64>)
    -> tensor<?x?xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
      {type = #hipsr.placeholder_type<barrier>} : tensor<?x?xf16>
      shape_region {
  ^bb0(%shape_ctx: !hipsr.context, %shape_input: tensor<?x3xf16>,
       %requested_shape: tensor<2xi64>):
    %input_shape = shape.shape_of %shape_input
        : tensor<?x3xf16> -> !shape.shape
    hipsr.shape_yield %input_shape : !shape.shape
  }
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
      outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
  return %result : tensor<?x?xf16>
}

// -----

// Canonicalization removes stale inputs from any position.
// CANONICALIZE-LABEL: func.func @middle_stale_input(
// CANONICALIZE: %[[INIT:.+]] = hipsr.placeholder
// CANONICALIZE-SAME: ins(%[[LHS:[^,]+]], %[[RHS:[^ )]+]] : tensor<4x8xf32>, tensor<4x8xf32>)
// CANONICALIZE-NEXT: %[[RESULT:.+]] = hipsr.add
// CANONICALIZE-SAME: ins(%[[LHS]], %[[RHS]] : tensor<4x8xf32>, tensor<4x8xf32>)
// CANONICALIZE-SAME: outs(%[[INIT]] : tensor<4x8xf32>)
func.func @middle_stale_input(
    %ctx: !hipsr.context, %lhs: tensor<4x8xf32>, %stale: tensor<1xi64>,
    %rhs: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %init = hipsr.placeholder(%ctx)
      ins(%lhs, %stale, %rhs
          : tensor<4x8xf32>, tensor<1xi64>, tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32>
  %result = hipsr.add(%ctx)
      ins(%lhs, %rhs : tensor<4x8xf32>, tensor<4x8xf32>)
      outs(%init : tensor<4x8xf32>) : tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// -----

// Absent regions are the valid staging form and preserve shape-graph chains.
// CHECK-LABEL: func.func @absent_region_chain(
// CHECK: %[[FIRST_INIT:.+]] = hipsr.placeholder
// CHECK-NOT: shape_region
// CHECK-NEXT: %[[FIRST:.+]] = hipsr.cast
// CHECK-NEXT: %[[SECOND_INIT:.+]] = hipsr.placeholder
// CHECK-SAME: ins(%[[FIRST_INIT]]
// CHECK-NOT: shape_region
// CHECK-NEXT: %[[SECOND:.+]] = hipsr.cast
func.func @absent_region_chain(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %first_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%first_init : tensor<4x8xf16>) : tensor<4x8xf16>
  %second_init = hipsr.placeholder(%ctx)
      ins(%first_init : tensor<4x8xf16>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32>
  %second = hipsr.cast(%ctx) ins(%first : tensor<4x8xf16>)
      outs(%second_init : tensor<4x8xf32>) : tensor<4x8xf32>
  return %second : tensor<4x8xf32>
}

// -----

// Arith and HIPSR constants are valid shape-graph roots.
// CHECK-LABEL: func.func @constant_shape_roots(
// CHECK: %[[ARITH:.+]] = arith.constant
// CHECK-NEXT: %[[HIPSR:.+]] = hipsr.constant
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder
// CHECK-SAME: ins(%[[ARITH]], %[[HIPSR]]
func.func @constant_shape_roots(
    %ctx: !hipsr.context, %input: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %arith_shape = arith.constant dense<[4, 8]> : tensor<2xi64>
  %hipsr_shape = hipsr.constant
      {value = dense<[4, 8]> : tensor<2xi64>} : tensor<2xi64>
  %init = hipsr.placeholder(%ctx)
      ins(%arith_shape, %hipsr_shape : tensor<2xi64>, tensor<2xi64>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %result = hipsr.add(%ctx)
      ins(%input, %input : tensor<4x8xf16>, tensor<4x8xf16>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// CSE must not merge placeholders tied to different consumers.
// CSE-LABEL: func.func @cse_keeps_placeholders(
// CSE: %[[LHS_INIT:.+]] = hipsr.placeholder
// CSE-NEXT: %[[RHS_INIT:.+]] = hipsr.placeholder
// CSE: outs(%[[LHS_INIT]]
// CSE: outs(%[[RHS_INIT]]
func.func @cse_keeps_placeholders(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>)
    -> (tensor<4x8xf16>, tensor<4x8xf16>) {
  %lhs_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %rhs_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %lhs_result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%lhs_init : tensor<4x8xf16>) : tensor<4x8xf16>
  %rhs_result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%rhs_init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %lhs_result, %rhs_result : tensor<4x8xf16>, tensor<4x8xf16>
}
