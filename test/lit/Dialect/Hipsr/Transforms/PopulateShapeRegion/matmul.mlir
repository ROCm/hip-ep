// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file -hipsr-populate-shape-region | FileCheck %s

// A 2-D MatMul only has to check that K agrees. Neither operand has batch
// dimensions, so nothing is broadcast. The result is M from A and N from B.
// CHECK-LABEL: func.func @matmul_2d(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<?x?xf16, #hipsr.mem<device>>, %[[B:.+]]: tensor<?x?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x?xf16, #hipsr.mem<device>>, tensor<?x?xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[A_SHAPE:.+]]: tensor<2xindex>, %[[B_SHAPE:.+]]: tensor<2xindex>):
// CHECK-NEXT: %[[A_K_INDEX:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[A_K:.+]] = shape.get_extent %[[A_SHAPE]], %[[A_K_INDEX]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[A_K_EXTENTS:.+]] = tensor.from_elements %[[A_K]] : tensor<1xindex>
// CHECK-NEXT: %[[B_K_INDEX:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[B_K:.+]] = shape.get_extent %[[B_SHAPE]], %[[B_K_INDEX]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[B_K_EXTENTS:.+]] = tensor.from_elements %[[B_K]] : tensor<1xindex>
// CHECK-NEXT: %[[K_WITNESS:.+]] = shape.cstr_eq %[[A_K_EXTENTS]], %[[B_K_EXTENTS]] : tensor<1xindex>, tensor<1xindex>
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[K_WITNESS]] -> (tensor<2xindex>) {
// CHECK-NEXT: %[[M_INDEX:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[M:.+]] = shape.get_extent %[[A_SHAPE]], %[[M_INDEX]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[N_INDEX:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[N:.+]] = shape.get_extent %[[B_SHAPE]], %[[N_INDEX]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[OUTPUT_SHAPE:.+]] = tensor.from_elements %[[M]], %[[N]] : tensor<2xindex>
// CHECK-NEXT: shape.assuming_yield %[[OUTPUT_SHAPE]] : tensor<2xindex>
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : tensor<2xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x?xf16, #hipsr.mem<device>>, tensor<?x?xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x?xf16, #hipsr.mem<device>>) : tensor<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @matmul_2d(%ctx: !hipsr.context, %a: tensor<?x?xf16, #hipsr.mem<device>>,
                     %b: tensor<?x?xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?x?xf16, #hipsr.mem<device>>, tensor<?x?xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?xf16, #hipsr.mem<device>>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?x?xf16, #hipsr.mem<device>>, tensor<?x?xf16, #hipsr.mem<device>>)
      outs(%init : tensor<?x?xf16, #hipsr.mem<device>>) : tensor<?x?xf16, #hipsr.mem<device>>
  return
}

// -----

// A vector A contributes no M dimension, so the assuming region reads only N.
// CHECK-LABEL: func.func @matmul_vector_matrix(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<?xf16, #hipsr.mem<device>>, %[[B:.+]]: tensor<?x?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?xf16, #hipsr.mem<device>>, tensor<?x?xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[A_SHAPE:.+]]: tensor<1xindex>, %[[B_SHAPE:.+]]: tensor<2xindex>):
// CHECK-NEXT: %[[A_K_INDEX:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[A_K:.+]] = shape.get_extent %[[A_SHAPE]], %[[A_K_INDEX]] : tensor<1xindex>, index -> index
// CHECK-NEXT: %[[A_K_EXTENTS:.+]] = tensor.from_elements %[[A_K]] : tensor<1xindex>
// CHECK-NEXT: %[[B_K_INDEX:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[B_K:.+]] = shape.get_extent %[[B_SHAPE]], %[[B_K_INDEX]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[B_K_EXTENTS:.+]] = tensor.from_elements %[[B_K]] : tensor<1xindex>
// CHECK-NEXT: %[[K_WITNESS:.+]] = shape.cstr_eq %[[A_K_EXTENTS]], %[[B_K_EXTENTS]] : tensor<1xindex>, tensor<1xindex>
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[K_WITNESS]] -> (tensor<1xindex>) {
// CHECK-NEXT: %[[N_INDEX:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[N:.+]] = shape.get_extent %[[B_SHAPE]], %[[N_INDEX]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[OUTPUT_SHAPE:.+]] = tensor.from_elements %[[N]] : tensor<1xindex>
// CHECK-NEXT: shape.assuming_yield %[[OUTPUT_SHAPE]] : tensor<1xindex>
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : tensor<1xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?xf16, #hipsr.mem<device>>, tensor<?x?xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?xf16, #hipsr.mem<device>>) : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @matmul_vector_matrix(%ctx: !hipsr.context, %a: tensor<?xf16, #hipsr.mem<device>>,
                                %b: tensor<?x?xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?xf16, #hipsr.mem<device>>, tensor<?x?xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16, #hipsr.mem<device>>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?xf16, #hipsr.mem<device>>, tensor<?x?xf16, #hipsr.mem<device>>)
      outs(%init : tensor<?xf16, #hipsr.mem<device>>) : tensor<?xf16, #hipsr.mem<device>>
  return
}

// -----

// A vector B contributes no N dimension, so the assuming region reads only M.
// CHECK-LABEL: func.func @matmul_matrix_vector(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<?x?xf16, #hipsr.mem<device>>, %[[B:.+]]: tensor<?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[A_SHAPE:.+]]: tensor<2xindex>, %[[B_SHAPE:.+]]: tensor<1xindex>):
// CHECK-NEXT: %[[A_K_INDEX:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[A_K:.+]] = shape.get_extent %[[A_SHAPE]], %[[A_K_INDEX]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[A_K_EXTENTS:.+]] = tensor.from_elements %[[A_K]] : tensor<1xindex>
// CHECK-NEXT: %[[B_K_INDEX:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[B_K:.+]] = shape.get_extent %[[B_SHAPE]], %[[B_K_INDEX]] : tensor<1xindex>, index -> index
// CHECK-NEXT: %[[B_K_EXTENTS:.+]] = tensor.from_elements %[[B_K]] : tensor<1xindex>
// CHECK-NEXT: %[[K_WITNESS:.+]] = shape.cstr_eq %[[A_K_EXTENTS]], %[[B_K_EXTENTS]] : tensor<1xindex>, tensor<1xindex>
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[K_WITNESS]] -> (tensor<1xindex>) {
// CHECK-NEXT: %[[M_INDEX:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[M:.+]] = shape.get_extent %[[A_SHAPE]], %[[M_INDEX]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[OUTPUT_SHAPE:.+]] = tensor.from_elements %[[M]] : tensor<1xindex>
// CHECK-NEXT: shape.assuming_yield %[[OUTPUT_SHAPE]] : tensor<1xindex>
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : tensor<1xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?xf16, #hipsr.mem<device>>) : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @matmul_matrix_vector(%ctx: !hipsr.context, %a: tensor<?x?xf16, #hipsr.mem<device>>,
                                %b: tensor<?xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?x?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16, #hipsr.mem<device>>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?x?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>)
      outs(%init : tensor<?xf16, #hipsr.mem<device>>) : tensor<?xf16, #hipsr.mem<device>>
  return
}

// -----

// Two vectors leave no extent at all, so the assuming region reads nothing and
// the scalar result gets its rank-0 extent tensor from a constant.
// CHECK-LABEL: func.func @matmul_dot(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<?xf16, #hipsr.mem<device>>, %[[B:.+]]: tensor<?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<f16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[A_SHAPE:.+]]: tensor<1xindex>, %[[B_SHAPE:.+]]: tensor<1xindex>):
// CHECK-NEXT: %[[A_K_INDEX:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[A_K:.+]] = shape.get_extent %[[A_SHAPE]], %[[A_K_INDEX]] : tensor<1xindex>, index -> index
// CHECK-NEXT: %[[A_K_EXTENTS:.+]] = tensor.from_elements %[[A_K]] : tensor<1xindex>
// CHECK-NEXT: %[[B_K_INDEX:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[B_K:.+]] = shape.get_extent %[[B_SHAPE]], %[[B_K_INDEX]] : tensor<1xindex>, index -> index
// CHECK-NEXT: %[[B_K_EXTENTS:.+]] = tensor.from_elements %[[B_K]] : tensor<1xindex>
// CHECK-NEXT: %[[K_WITNESS:.+]] = shape.cstr_eq %[[A_K_EXTENTS]], %[[B_K_EXTENTS]] : tensor<1xindex>, tensor<1xindex>
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[K_WITNESS]] -> (tensor<0xindex>) {
// CHECK-NEXT: %[[OUTPUT_SHAPE:.+]] = arith.constant dense<> : tensor<0xindex>
// CHECK-NEXT: shape.assuming_yield %[[OUTPUT_SHAPE]] : tensor<0xindex>
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : tensor<0xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<f16, #hipsr.mem<device>>) : tensor<f16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @matmul_dot(%ctx: !hipsr.context, %a: tensor<?xf16, #hipsr.mem<device>>,
                      %b: tensor<?xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<f16, #hipsr.mem<device>>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>)
      outs(%init : tensor<f16, #hipsr.mem<device>>) : tensor<f16, #hipsr.mem<device>>
  return
}

// -----

// The recipe right-aligns the rank-2 and rank-1 batch prefixes and broadcasts
// them. It then reads the result back one extent at a time to lead the result
// shape. The broadcast rank is known, so that read-back is a compile-time loop.
// CHECK-LABEL: func.func @matmul_batched(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<?x1x?x?xf16, #hipsr.mem<device>>, %[[B:.+]]: tensor<3x?x?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x1x?x?xf16, #hipsr.mem<device>>, tensor<3x?x?xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x3x?x?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[A_SHAPE:.+]]: tensor<4xindex>, %[[B_SHAPE:.+]]: tensor<3xindex>):
// CHECK-NEXT: %[[A_K_INDEX:.+]] = arith.constant 3 : index
// CHECK-NEXT: %[[A_K:.+]] = shape.get_extent %[[A_SHAPE]], %[[A_K_INDEX]] : tensor<4xindex>, index -> index
// CHECK-NEXT: %[[A_K_EXTENTS:.+]] = tensor.from_elements %[[A_K]] : tensor<1xindex>
// CHECK-NEXT: %[[B_K_INDEX:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[B_K:.+]] = shape.get_extent %[[B_SHAPE]], %[[B_K_INDEX]] : tensor<3xindex>, index -> index
// CHECK-NEXT: %[[B_K_EXTENTS:.+]] = tensor.from_elements %[[B_K]] : tensor<1xindex>
// CHECK-NEXT: %[[K_WITNESS:.+]] = shape.cstr_eq %[[A_K_EXTENTS]], %[[B_K_EXTENTS]] : tensor<1xindex>, tensor<1xindex>
// CHECK-NEXT: %[[A_BATCH_D0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[A_BATCH_E0:.+]] = shape.get_extent %[[A_SHAPE]], %[[A_BATCH_D0]] : tensor<4xindex>, index -> index
// CHECK-NEXT: %[[A_BATCH_D1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[A_BATCH_E1:.+]] = shape.get_extent %[[A_SHAPE]], %[[A_BATCH_D1]] : tensor<4xindex>, index -> index
// CHECK-NEXT: %[[A_BATCH:.+]] = tensor.from_elements %[[A_BATCH_E0]], %[[A_BATCH_E1]] : tensor<2xindex>
// CHECK-NEXT: %[[B_BATCH_D0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[B_BATCH_E0:.+]] = shape.get_extent %[[B_SHAPE]], %[[B_BATCH_D0]] : tensor<3xindex>, index -> index
// CHECK-NEXT: %[[B_BATCH:.+]] = tensor.from_elements %[[B_BATCH_E0]] : tensor<1xindex>
// CHECK-NEXT: %[[BATCH_WITNESS:.+]] = shape.cstr_broadcastable %[[A_BATCH]], %[[B_BATCH]] : tensor<2xindex>, tensor<1xindex>
// CHECK-NEXT: %[[WITNESS:.+]] = shape.assuming_all %[[K_WITNESS]], %[[BATCH_WITNESS]]
// CHECK-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (tensor<4xindex>) {
// CHECK-NEXT: %[[BATCH:.+]] = shape.broadcast %[[A_BATCH]], %[[B_BATCH]] : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
// CHECK-NEXT: %[[BATCH_D0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[BATCH_E0:.+]] = shape.get_extent %[[BATCH]], %[[BATCH_D0]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[BATCH_D1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[BATCH_E1:.+]] = shape.get_extent %[[BATCH]], %[[BATCH_D1]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[M_INDEX:.+]] = arith.constant 2 : index
// CHECK-NEXT: %[[M:.+]] = shape.get_extent %[[A_SHAPE]], %[[M_INDEX]] : tensor<4xindex>, index -> index
// CHECK-NEXT: %[[N_INDEX:.+]] = arith.constant 2 : index
// CHECK-NEXT: %[[N:.+]] = shape.get_extent %[[B_SHAPE]], %[[N_INDEX]] : tensor<3xindex>, index -> index
// CHECK-NEXT: %[[OUTPUT_SHAPE:.+]] = tensor.from_elements %[[BATCH_E0]], %[[BATCH_E1]], %[[M]], %[[N]] : tensor<4xindex>
// CHECK-NEXT: shape.assuming_yield %[[OUTPUT_SHAPE]] : tensor<4xindex>
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : tensor<4xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x1x?x?xf16, #hipsr.mem<device>>, tensor<3x?x?xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x3x?x?xf16, #hipsr.mem<device>>) : tensor<?x3x?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @matmul_batched(%ctx: !hipsr.context,
                          %a: tensor<?x1x?x?xf16, #hipsr.mem<device>>,
                          %b: tensor<3x?x?xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?x1x?x?xf16, #hipsr.mem<device>>, tensor<3x?x?xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<?x3x?x?xf16, #hipsr.mem<device>>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?x1x?x?xf16, #hipsr.mem<device>>, tensor<3x?x?xf16, #hipsr.mem<device>>)
      outs(%init : tensor<?x3x?x?xf16, #hipsr.mem<device>>) : tensor<?x3x?x?xf16, #hipsr.mem<device>>
  return
}
