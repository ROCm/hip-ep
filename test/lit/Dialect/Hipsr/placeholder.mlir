// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file -cse %s | FileCheck %s --check-prefix=CSE

// Dynamic, static, and rank-0 results parse and print together.
// CHECK-LABEL: func.func @tensor_forms(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME: %[[DYNAMIC_INPUT:.*]]: tensor<?x?xf32>,
// CHECK-SAME: %[[STATIC_INPUT:.*]]: tensor<4x8xf32>,
// CHECK-SAME: %[[SCALAR_INPUT:.*]]: tensor<f32>)
// CHECK-SAME: -> (tensor<?x?xf16>, tensor<4x8xi64>, tensor<f16>) {
// CHECK-NEXT: %[[DYNAMIC_INIT:.*]] = hipsr.placeholder : tensor<?x?xf16>
// CHECK-NEXT: %[[DYNAMIC_RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[DYNAMIC_INPUT]] : tensor<?x?xf32>) outs(%[[DYNAMIC_INIT]] : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK-NEXT: %[[STATIC_INIT:.*]] = hipsr.placeholder : tensor<4x8xi64>
// CHECK-NEXT: %[[STATIC_RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[STATIC_INPUT]] : tensor<4x8xf32>) outs(%[[STATIC_INIT]] : tensor<4x8xi64>) : tensor<4x8xi64>
// CHECK-NEXT: %[[SCALAR_INIT:.*]] = hipsr.placeholder : tensor<f16>
// CHECK-NEXT: %[[SCALAR_RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[SCALAR_INPUT]] : tensor<f32>) outs(%[[SCALAR_INIT]] : tensor<f16>) : tensor<f16>
// CHECK-NEXT: return %[[DYNAMIC_RESULT]], %[[STATIC_RESULT]], %[[SCALAR_RESULT]] : tensor<?x?xf16>, tensor<4x8xi64>, tensor<f16>
// CHECK-NEXT: }
func.func @tensor_forms(
    %ctx: !hipsr.context, %dynamic_input: tensor<?x?xf32>,
    %static_input: tensor<4x8xf32>, %scalar_input: tensor<f32>)
    -> (tensor<?x?xf16>, tensor<4x8xi64>, tensor<f16>) {
  %dynamic_init = hipsr.placeholder : tensor<?x?xf16>
  %dynamic_result = hipsr.cast(%ctx) ins(%dynamic_input : tensor<?x?xf32>)
      outs(%dynamic_init : tensor<?x?xf16>) : tensor<?x?xf16>
  %static_init = hipsr.placeholder : tensor<4x8xi64>
  %static_result = hipsr.cast(%ctx) ins(%static_input : tensor<4x8xf32>)
      outs(%static_init : tensor<4x8xi64>) : tensor<4x8xi64>
  %scalar_init = hipsr.placeholder : tensor<f16>
  %scalar_result = hipsr.cast(%ctx) ins(%scalar_input : tensor<f32>)
      outs(%scalar_init : tensor<f16>) : tensor<f16>
  return %dynamic_result, %static_result, %scalar_result
      : tensor<?x?xf16>, tensor<4x8xi64>, tensor<f16>
}

// -----
// CSE must not merge placeholders used by different DPS ops.
// CSE-LABEL: func.func @cse_keeps_placeholders(
// CSE-SAME: %[[CTX:.*]]: !hipsr.context, %[[LHS:.*]]: tensor<4x8xf32>,
// CSE-SAME: %[[RHS:.*]]: tensor<4x8xf32>)
// CSE-SAME: -> (tensor<4x8xf16>, tensor<4x8xf16>) {
// CSE-NEXT: %[[LHS_INIT:.*]] = hipsr.placeholder : tensor<4x8xf16>
// CSE-NEXT: %[[RHS_INIT:.*]] = hipsr.placeholder : tensor<4x8xf16>
// CSE-NEXT: %[[LHS_RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[LHS]] : tensor<4x8xf32>) outs(%[[LHS_INIT]] : tensor<4x8xf16>) : tensor<4x8xf16>
// CSE-NEXT: %[[RHS_RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[RHS]] : tensor<4x8xf32>) outs(%[[RHS_INIT]] : tensor<4x8xf16>) : tensor<4x8xf16>
// CSE-NEXT: return %[[LHS_RESULT]], %[[RHS_RESULT]] : tensor<4x8xf16>, tensor<4x8xf16>
// CSE-NEXT: }
func.func @cse_keeps_placeholders(
    %ctx: !hipsr.context, %lhs: tensor<4x8xf32>, %rhs: tensor<4x8xf32>)
    -> (tensor<4x8xf16>, tensor<4x8xf16>) {
  %lhs_init = hipsr.placeholder : tensor<4x8xf16>
  %rhs_init = hipsr.placeholder : tensor<4x8xf16>
  %lhs_result = hipsr.cast(%ctx) ins(%lhs : tensor<4x8xf32>)
      outs(%lhs_init : tensor<4x8xf16>) : tensor<4x8xf16>
  %rhs_result = hipsr.cast(%ctx) ins(%rhs : tensor<4x8xf32>)
      outs(%rhs_init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %lhs_result, %rhs_result : tensor<4x8xf16>, tensor<4x8xf16>
}
