// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file -hipsr-populate-shape-region %s | FileCheck %s --check-prefix=POPULATE

// Post-bufferization memref syntax remains supported by the compute op.
// CHECK-LABEL: func.func @add_memref(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[LHS:.+]]: memref<4x1024xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[RHS:.+]]: memref<1024xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[INIT:.+]]: memref<4x1024xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: hipsr.add(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<1024xf16, #hipsr.mem<device>>) outs(%[[INIT]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @add_memref(
    %ctx: !hipsr.context,
    %lhs: memref<4x1024xf16, #hipsr.mem<device>>,
    %rhs: memref<1024xf16, #hipsr.mem<device>>,
    %init: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.add(%ctx)
      ins(%lhs, %rhs : memref<4x1024xf16, #hipsr.mem<device>>,
                          memref<1024xf16, #hipsr.mem<device>>)
      outs(%init : memref<4x1024xf16, #hipsr.mem<device>>)
  return
}

// -----

// Add broadcasts the two normal placeholder shape arguments.
// POPULATE-LABEL: func.func @add_broadcast(
// POPULATE-SAME: %[[CTX:.+]]: !hipsr.context, %[[LHS:.+]]: tensor<?x1024xf16>, %[[RHS:.+]]: tensor<1024xf16>) -> tensor<?x1024xf16> {
// POPULATE-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<?x1024xf16>, tensor<1024xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<?x1024xf16> shape_region {
// POPULATE-NEXT: ^bb0(%[[LHS_SHAPE:.+]]: !shape.shape, %[[RHS_SHAPE:.+]]: !shape.shape):
// POPULATE-NEXT: %[[RESULT_SHAPE:.+]] = shape.broadcast %[[LHS_SHAPE]], %[[RHS_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: %[[RESULT:.+]] = hipsr.add(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<?x1024xf16>, tensor<1024xf16>) outs(%[[INIT]] : tensor<?x1024xf16>) : tensor<?x1024xf16>
// POPULATE-NEXT: return %[[RESULT]] : tensor<?x1024xf16>
// POPULATE-NEXT: }
func.func @add_broadcast(
    %ctx: !hipsr.context, %lhs: tensor<?x1024xf16>,
    %rhs: tensor<1024xf16>) -> tensor<?x1024xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%lhs, %rhs : tensor<?x1024xf16>, tensor<1024xf16>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x1024xf16>
  %result = hipsr.add(%ctx)
      ins(%lhs, %rhs : tensor<?x1024xf16>, tensor<1024xf16>)
      outs(%init : tensor<?x1024xf16>) : tensor<?x1024xf16>
  return %result : tensor<?x1024xf16>
}
