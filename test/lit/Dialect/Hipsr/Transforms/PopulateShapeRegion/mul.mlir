// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-populate-shape-region | FileCheck %s

// Mul broadcasts its two input shapes into the placeholder result shape.
// CHECK-LABEL: func.func @mul_normal(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[LHS:.+]]: tensor<?x1024xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[RHS:.+]]: tensor<1024xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<?x1024xf16, #hipsr.mem<device>>, tensor<1024xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x1024xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[LHS_SHAPE:.+]]: tensor<2xindex>, %[[RHS_SHAPE:.+]]: tensor<1xindex>):
// CHECK-NEXT: %[[BROADCAST:.+]] = shape.broadcast %[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
// CHECK-NEXT: hipsr.shape_yield %[[BROADCAST]] : tensor<2xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.mul(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<?x1024xf16, #hipsr.mem<device>>, tensor<1024xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x1024xf16, #hipsr.mem<device>>) : tensor<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @mul_normal(
    %ctx: !hipsr.context, %lhs: tensor<?x1024xf16, #hipsr.mem<device>>,
    %rhs: tensor<1024xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%lhs, %rhs : tensor<?x1024xf16, #hipsr.mem<device>>, tensor<1024xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<?x1024xf16, #hipsr.mem<device>>
  %result = hipsr.mul(%ctx)
      ins(%lhs, %rhs : tensor<?x1024xf16, #hipsr.mem<device>>, tensor<1024xf16, #hipsr.mem<device>>)
      outs(%init : tensor<?x1024xf16, #hipsr.mem<device>>) : tensor<?x1024xf16, #hipsr.mem<device>>
  return
}
