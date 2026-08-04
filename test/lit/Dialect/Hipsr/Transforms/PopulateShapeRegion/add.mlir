// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-populate-shape-region | FileCheck %s

// Add broadcasts its two input shapes into the placeholder result shape.
// CHECK-LABEL: func.func @add_normal(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[LHS:.+]]: tensor<?x1024xf16>,
// CHECK-SAME: %[[RHS:.+]]: tensor<1024xf16>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<?x1024xf16>, tensor<1024xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x1024xf16> shape_region {
// CHECK-NEXT: ^bb0(%[[LHS_SHAPE:.+]]: !shape.shape, %[[RHS_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: %[[BROADCAST:.+]] = shape.broadcast %[[LHS_SHAPE]], %[[RHS_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: hipsr.shape_yield2 %[[BROADCAST]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.add(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<?x1024xf16>, tensor<1024xf16>) outs(%[[INIT]] : tensor<?x1024xf16>) : tensor<?x1024xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @add_normal(
    %ctx: !hipsr.context, %lhs: tensor<?x1024xf16>,
    %rhs: tensor<1024xf16>) {
  %init = hipsr.placeholder(%ctx)
      ins(%lhs, %rhs : tensor<?x1024xf16>, tensor<1024xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<?x1024xf16>
  %result = hipsr.add(%ctx)
      ins(%lhs, %rhs : tensor<?x1024xf16>, tensor<1024xf16>)
      outs(%init : tensor<?x1024xf16>) : tensor<?x1024xf16>
  return
}
