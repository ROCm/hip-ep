// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file -cse %s | FileCheck %s --check-prefix=CSE

// -----
// A dynamic tensor result parses and prints.
// CHECK-LABEL: func.func @single_dynamic(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<?x?xf32>) -> tensor<?x?xf16> {
// CHECK-NEXT: %[[INIT:.*]] = hipsr.placeholder : tensor<?x?xf16>
// CHECK-NEXT: %[[RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK-NEXT: return %[[RESULT]] : tensor<?x?xf16>
// CHECK-NEXT: }
func.func @single_dynamic(%ctx: !hipsr.context,
                          %input: tensor<?x?xf32>) -> tensor<?x?xf16> {
  %init = hipsr.placeholder : tensor<?x?xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<?x?xf32>)
      outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
  return %result : tensor<?x?xf16>
}

// -----
// A static tensor result parses and prints.
// CHECK-LABEL: func.func @single_static(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4x8xf32>) -> tensor<4x8xi64> {
// CHECK-NEXT: %[[INIT:.*]] = hipsr.placeholder : tensor<4x8xi64>
// CHECK-NEXT: %[[RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<4x8xf32>) outs(%[[INIT]] : tensor<4x8xi64>) : tensor<4x8xi64>
// CHECK-NEXT: return %[[RESULT]] : tensor<4x8xi64>
// CHECK-NEXT: }
func.func @single_static(%ctx: !hipsr.context,
                         %input: tensor<4x8xf32>) -> tensor<4x8xi64> {
  %init = hipsr.placeholder : tensor<4x8xi64>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xi64>) : tensor<4x8xi64>
  return %result : tensor<4x8xi64>
}

// -----
// A rank-0 tensor result parses and prints.
// CHECK-LABEL: func.func @rank0(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<f32>) -> tensor<f16> {
// CHECK-NEXT: %[[INIT:.*]] = hipsr.placeholder : tensor<f16>
// CHECK-NEXT: %[[RESULT:.*]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<f32>) outs(%[[INIT]] : tensor<f16>) : tensor<f16>
// CHECK-NEXT: return %[[RESULT]] : tensor<f16>
// CHECK-NEXT: }
func.func @rank0(%ctx: !hipsr.context,
                 %input: tensor<f32>) -> tensor<f16> {
  %init = hipsr.placeholder : tensor<f16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<f32>)
      outs(%init : tensor<f16>) : tensor<f16>
  return %result : tensor<f16>
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
