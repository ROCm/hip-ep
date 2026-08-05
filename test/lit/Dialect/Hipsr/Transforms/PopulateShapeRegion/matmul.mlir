// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file -hipsr-populate-shape-region | FileCheck %s

// A 2-D MatMul checks K equality, broadcasts empty batch shapes, and appends
// M from A and N from B.
// CHECK-LABEL: func.func @matmul_2d(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<?x?xf16>, %[[B:.+]]: tensor<?x?xf16>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x?xf16>, tensor<?x?xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?xf16> shape_region {
// CHECK-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: %[[A_K_INDEX:.+]] = shape.const_size 1
// CHECK-NEXT: %[[A_K:.+]] = shape.get_extent %[[A_SHAPE]], %[[A_K_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT: %[[A_K_SHAPE:.+]] = shape.from_extents %[[A_K]] : !shape.size
// CHECK-NEXT: %[[B_K_INDEX:.+]] = shape.const_size 0
// CHECK-NEXT: %[[B_K:.+]] = shape.get_extent %[[B_SHAPE]], %[[B_K_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT: %[[B_K_SHAPE:.+]] = shape.from_extents %[[B_K]] : !shape.size
// CHECK-NEXT: %[[K_WITNESS:.+]] = shape.cstr_eq %[[A_K_SHAPE]], %[[B_K_SHAPE]] : !shape.shape, !shape.shape
// CHECK-NEXT: %[[A_BATCH_INDEX:.+]] = shape.const_size 0
// CHECK-NEXT: %[[A_BATCH:.+]], %[[A_SUFFIX:.+]] = "shape.split_at"(%[[A_SHAPE]], %[[A_BATCH_INDEX]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT: %[[B_BATCH_INDEX:.+]] = shape.const_size 0
// CHECK-NEXT: %[[B_BATCH:.+]], %[[B_SUFFIX:.+]] = "shape.split_at"(%[[B_SHAPE]], %[[B_BATCH_INDEX]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT: %[[BATCH_WITNESS:.+]] = shape.cstr_broadcastable %[[A_BATCH]], %[[B_BATCH]] : !shape.shape, !shape.shape
// CHECK-NEXT: %[[WITNESS:.+]] = shape.assuming_all %[[K_WITNESS]], %[[BATCH_WITNESS]]
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// CHECK-NEXT: %[[BATCH_SHAPE:.+]] = shape.broadcast %[[A_BATCH]], %[[B_BATCH]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: %[[M_INDEX:.+]] = shape.const_size 0
// CHECK-NEXT: %[[M:.+]] = shape.get_extent %[[A_SHAPE]], %[[M_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT: %[[N_INDEX:.+]] = shape.const_size 1
// CHECK-NEXT: %[[N:.+]] = shape.get_extent %[[B_SHAPE]], %[[N_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT: %[[MATRIX_SHAPE:.+]] = shape.from_extents %[[M]], %[[N]] : !shape.size, !shape.size
// CHECK-NEXT: %[[OUTPUT_SHAPE:.+]] = shape.concat %[[BATCH_SHAPE]], %[[MATRIX_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: shape.assuming_yield %[[OUTPUT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x?xf16>, tensor<?x?xf16>) outs(%[[INIT]] : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @matmul_2d(%ctx: !hipsr.context, %a: tensor<?x?xf16>,
                     %b: tensor<?x?xf16>) {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?x?xf16>, tensor<?x?xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?xf16>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
  return
}

// -----

// A vector A contributes no M dimension, leaving only N after empty-batch
// broadcasting.
// CHECK-LABEL: func.func @matmul_vector_matrix(
// CHECK: %[[INIT:.+]] = hipsr.placeholder
// CHECK-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// CHECK: %[[K_WITNESS:.+]] = shape.cstr_eq
// CHECK: %[[BATCH_WITNESS:.+]] = shape.cstr_broadcastable %[[A_BATCH:.+]], %[[B_BATCH:.+]] : !shape.shape, !shape.shape
// CHECK-NEXT: %[[WITNESS:.+]] = shape.assuming_all %[[K_WITNESS]], %[[BATCH_WITNESS]]
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// CHECK-NEXT: %[[BROADCAST:.+]] = shape.broadcast %[[A_BATCH]], %[[B_BATCH]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NOT: shape.get_extent %[[A_SHAPE]]
// CHECK-NEXT: %[[N_INDEX:.+]] = shape.const_size 1
// CHECK-NEXT: %[[N:.+]] = shape.get_extent %[[B_SHAPE]], %[[N_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT: %[[MATRIX_SHAPE:.+]] = shape.from_extents %[[N]] : !shape.size
// CHECK-NEXT: %[[OUTPUT_SHAPE:.+]] = shape.concat %[[BROADCAST]], %[[MATRIX_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: shape.assuming_yield %[[OUTPUT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.matmul
// CHECK-NOT: shape_region
// CHECK-NEXT: return
func.func @matmul_vector_matrix(%ctx: !hipsr.context, %a: tensor<?xf16>,
                                %b: tensor<?x?xf16>) {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?xf16>, tensor<?x?xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?xf16>, tensor<?x?xf16>)
      outs(%init : tensor<?xf16>) : tensor<?xf16>
  return
}

// -----

// A vector B contributes no N dimension, leaving only M after empty-batch
// broadcasting.
// CHECK-LABEL: func.func @matmul_matrix_vector(
// CHECK: %[[INIT:.+]] = hipsr.placeholder
// CHECK-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// CHECK: %[[K_WITNESS:.+]] = shape.cstr_eq
// CHECK: %[[BATCH_WITNESS:.+]] = shape.cstr_broadcastable %[[A_BATCH:.+]], %[[B_BATCH:.+]] : !shape.shape, !shape.shape
// CHECK-NEXT: %[[WITNESS:.+]] = shape.assuming_all %[[K_WITNESS]], %[[BATCH_WITNESS]]
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// CHECK-NEXT: %[[BROADCAST:.+]] = shape.broadcast %[[A_BATCH]], %[[B_BATCH]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: %[[M_INDEX:.+]] = shape.const_size 0
// CHECK-NEXT: %[[M:.+]] = shape.get_extent %[[A_SHAPE]], %[[M_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NOT: shape.get_extent %[[B_SHAPE]]
// CHECK-NEXT: %[[MATRIX_SHAPE:.+]] = shape.from_extents %[[M]] : !shape.size
// CHECK-NEXT: %[[OUTPUT_SHAPE:.+]] = shape.concat %[[BROADCAST]], %[[MATRIX_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: shape.assuming_yield %[[OUTPUT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.matmul
// CHECK-NOT: shape_region
// CHECK-NEXT: return
func.func @matmul_matrix_vector(%ctx: !hipsr.context, %a: tensor<?x?xf16>,
                                %b: tensor<?xf16>) {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?x?xf16>, tensor<?xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?x?xf16>, tensor<?xf16>)
      outs(%init : tensor<?xf16>) : tensor<?xf16>
  return
}

// -----

// Two vectors produce one empty shape value for the scalar result.
// CHECK-LABEL: func.func @matmul_dot(
// CHECK: %[[INIT:.+]] = hipsr.placeholder
// CHECK-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// CHECK: %[[K_WITNESS:.+]] = shape.cstr_eq
// CHECK: %[[BATCH_WITNESS:.+]] = shape.cstr_broadcastable %[[A_BATCH:.+]], %[[B_BATCH:.+]] : !shape.shape, !shape.shape
// CHECK-NEXT: %[[WITNESS:.+]] = shape.assuming_all %[[K_WITNESS]], %[[BATCH_WITNESS]]
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// CHECK-NEXT: %[[BROADCAST:.+]] = shape.broadcast %[[A_BATCH]], %[[B_BATCH]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NOT: shape.get_extent
// CHECK-NEXT: %[[EMPTY_SHAPE:.+]] = shape.from_extents{{ *}}:
// CHECK-NEXT: %[[OUTPUT_SHAPE:.+]] = shape.concat %[[BROADCAST]], %[[EMPTY_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: shape.assuming_yield %[[OUTPUT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.matmul
// CHECK-NOT: shape_region
// CHECK-NEXT: return
func.func @matmul_dot(%ctx: !hipsr.context, %a: tensor<?xf16>,
                      %b: tensor<?xf16>) {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<f16>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
      outs(%init : tensor<f16>) : tensor<f16>
  return
}

// -----

// Batch prefixes of ranks two and one are right-aligned and broadcast before
// the matrix dimensions are appended.
// CHECK-LABEL: func.func @matmul_batched(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<?x1x?x?xf16>, %[[B:.+]]: tensor<3x?x?xf16>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x1x?x?xf16>, tensor<3x?x?xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x3x?x?xf16> shape_region {
// CHECK-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// CHECK: %[[K_WITNESS:.+]] = shape.cstr_eq
// CHECK-NEXT: %[[A_BATCH_INDEX:.+]] = shape.const_size 2
// CHECK-NEXT: %[[A_BATCH:.+]], %[[A_SUFFIX:.+]] = "shape.split_at"(%[[A_SHAPE]], %[[A_BATCH_INDEX]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT: %[[B_BATCH_INDEX:.+]] = shape.const_size 1
// CHECK-NEXT: %[[B_BATCH:.+]], %[[B_SUFFIX:.+]] = "shape.split_at"(%[[B_SHAPE]], %[[B_BATCH_INDEX]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT: %[[BATCH_WITNESS:.+]] = shape.cstr_broadcastable %[[A_BATCH]], %[[B_BATCH]] : !shape.shape, !shape.shape
// CHECK-NEXT: %[[WITNESS:.+]] = shape.assuming_all %[[K_WITNESS]], %[[BATCH_WITNESS]]
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// CHECK-NEXT: %[[BROADCAST:.+]] = shape.broadcast %[[A_BATCH]], %[[B_BATCH]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: %[[M_INDEX:.+]] = shape.const_size 2
// CHECK-NEXT: %[[M:.+]] = shape.get_extent %[[A_SHAPE]], %[[M_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT: %[[N_INDEX:.+]] = shape.const_size 2
// CHECK-NEXT: %[[N:.+]] = shape.get_extent %[[B_SHAPE]], %[[N_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT: %[[MATRIX_SHAPE:.+]] = shape.from_extents %[[M]], %[[N]] : !shape.size, !shape.size
// CHECK-NEXT: %[[OUTPUT_SHAPE:.+]] = shape.concat %[[BROADCAST]], %[[MATRIX_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: shape.assuming_yield %[[OUTPUT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x1x?x?xf16>, tensor<3x?x?xf16>) outs(%[[INIT]] : tensor<?x3x?x?xf16>) : tensor<?x3x?x?xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @matmul_batched(%ctx: !hipsr.context,
                          %a: tensor<?x1x?x?xf16>,
                          %b: tensor<3x?x?xf16>) {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?x1x?x?xf16>, tensor<3x?x?xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<?x3x?x?xf16>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?x1x?x?xf16>, tensor<3x?x?xf16>)
      outs(%init : tensor<?x3x?x?xf16>) : tensor<?x3x?x?xf16>
  return
}
