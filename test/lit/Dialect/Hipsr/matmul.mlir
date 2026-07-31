// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics -hipsr-populate-shape-region %s | FileCheck %s --check-prefix=POPULATE

func.func @matmul_rank0_a(
    %ctx: !hipsr.context, %a: tensor<f16>, %b: tensor<4096x1024xf16>,
    %init: tensor<1024xf16>) -> tensor<1024xf16> {
  // expected-error@+1 {{operand A must be at least 1-D}}
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<f16>, tensor<4096x1024xf16>)
      outs(%init : tensor<1024xf16>) : tensor<1024xf16>
  return %result : tensor<1024xf16>
}

// -----

func.func @matmul_rank0_b(
    %ctx: !hipsr.context, %a: tensor<64x4096xf16>, %b: tensor<f16>,
    %init: tensor<64xf16>) -> tensor<64xf16> {
  // expected-error@+1 {{operand B must be at least 1-D}}
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<64x4096xf16>, tensor<f16>)
      outs(%init : tensor<64xf16>) : tensor<64xf16>
  return %result : tensor<64xf16>
}

// -----

// DPS init and result types must match.
func.func @matmul_init_result_mismatch(
    %ctx: !hipsr.context, %a: tensor<2x3x64x4096xf16>,
    %b: tensor<2x3x4096x1024xf16>,
    %init: tensor<2x3x64x1024xf16>) -> tensor<4x3x64x1024xf16> {
  // expected-error@+1 {{to match type of corresponding result}}
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<2x3x64x4096xf16>, tensor<2x3x4096x1024xf16>)
      outs(%init : tensor<2x3x64x1024xf16>) : tensor<4x3x64x1024xf16>
  return %result : tensor<4x3x64x1024xf16>
}

// -----

// Data operands must be ranked tensors or device memrefs.
func.func @matmul_operand_not_shaped(
    %ctx: !hipsr.context, %a: f16, %b: tensor<4096x1024xf16>,
    %init: tensor<64x1024xf16>) -> tensor<64x1024xf16> {
  // expected-error@+1 {{operand #1 must be ranked tensor or device memref}}
  %result = "hipsr.matmul"(%ctx, %a, %b, %init)
      : (!hipsr.context, f16, tensor<4096x1024xf16>,
         tensor<64x1024xf16>) -> tensor<64x1024xf16>
  return %result : tensor<64x1024xf16>
}

// -----

// The normal placeholder receives two typed shapes. The recipe promotes
// vectors, checks contraction and batch compatibility, then computes one
// result shape under the combined witness.
// POPULATE-LABEL: func.func @matmul_2d(
// POPULATE-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<?x4096xf16>, %[[B:.+]]: tensor<4096x1024xf16>) -> tensor<?x1024xf16> {
// POPULATE-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x4096xf16>, tensor<4096x1024xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<?x1024xf16> shape_region {
// POPULATE-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// POPULATE-NEXT: %[[ONE:.+]] = arith.constant 1 : index
// POPULATE-NEXT: %[[MINUS_TWO:.+]] = arith.constant -2 : index
// POPULATE-NEXT: %[[A_RANK_SIZE:.+]] = shape.rank %[[A_SHAPE]] : !shape.shape -> !shape.size
// POPULATE-NEXT: %[[A_RANK:.+]] = shape.size_to_index %[[A_RANK_SIZE]] : !shape.size
// POPULATE-NEXT: %[[A_IS_VECTOR:.+]] = arith.cmpi eq, %[[A_RANK]], %[[ONE]] : index
// POPULATE-NEXT: %[[B_RANK_SIZE:.+]] = shape.rank %[[B_SHAPE]] : !shape.shape -> !shape.size
// POPULATE-NEXT: %[[B_RANK:.+]] = shape.size_to_index %[[B_RANK_SIZE]] : !shape.size
// POPULATE-NEXT: %[[B_IS_VECTOR:.+]] = arith.cmpi eq, %[[B_RANK]], %[[ONE]] : index
// POPULATE-NEXT: %[[EMPTY_SHAPE:.+]] = shape.from_extents  :
// POPULATE-NEXT: %[[ONE_SHAPE:.+]] = shape.from_extents %[[ONE]] : index
// POPULATE-NEXT: %[[A_PROMOTED:.+]] = shape.concat %[[ONE_SHAPE]], %[[A_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: %[[B_PROMOTED:.+]] = shape.concat %[[B_SHAPE]], %[[ONE_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: %[[A_NORMALIZED:.+]] = arith.select %[[A_IS_VECTOR]], %[[A_PROMOTED]], %[[A_SHAPE]] : !shape.shape
// POPULATE-NEXT: %[[B_NORMALIZED:.+]] = arith.select %[[B_IS_VECTOR]], %[[B_PROMOTED]], %[[B_SHAPE]] : !shape.shape
// POPULATE-NEXT: %[[A_BATCH:.+]], %[[A_MATRIX:.+]] = "shape.split_at"(%[[A_NORMALIZED]], %[[MINUS_TWO]]) : (!shape.shape, index) -> (!shape.shape, !shape.shape)
// POPULATE-NEXT: %[[B_BATCH:.+]], %[[B_MATRIX:.+]] = "shape.split_at"(%[[B_NORMALIZED]], %[[MINUS_TWO]]) : (!shape.shape, index) -> (!shape.shape, !shape.shape)
// POPULATE-NEXT: %[[M:.+]], %[[K_A:.+]] = "shape.split_at"(%[[A_MATRIX]], %[[ONE]]) : (!shape.shape, index) -> (!shape.shape, !shape.shape)
// POPULATE-NEXT: %[[K_B:.+]], %[[N:.+]] = "shape.split_at"(%[[B_MATRIX]], %[[ONE]]) : (!shape.shape, index) -> (!shape.shape, !shape.shape)
// POPULATE-NEXT: %[[K_WITNESS:.+]] = shape.cstr_eq %[[K_A]], %[[K_B]] : !shape.shape, !shape.shape
// POPULATE-NEXT: %[[BATCH_WITNESS:.+]] = shape.cstr_broadcastable %[[A_BATCH]], %[[B_BATCH]] : !shape.shape, !shape.shape
// POPULATE-NEXT: %[[WITNESS:.+]] = shape.assuming_all %[[K_WITNESS]], %[[BATCH_WITNESS]]
// POPULATE-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// POPULATE-NEXT: %[[BATCH:.+]] = shape.broadcast %[[A_BATCH]], %[[B_BATCH]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: %[[RESULT_M:.+]] = arith.select %[[A_IS_VECTOR]], %[[EMPTY_SHAPE]], %[[M]] : !shape.shape
// POPULATE-NEXT: %[[RESULT_N:.+]] = arith.select %[[B_IS_VECTOR]], %[[EMPTY_SHAPE]], %[[N]] : !shape.shape
// POPULATE-NEXT: %[[MATRIX:.+]] = shape.concat %[[RESULT_M]], %[[RESULT_N]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: %[[FINAL_SHAPE:.+]] = shape.concat %[[BATCH]], %[[MATRIX]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: shape.assuming_yield %[[FINAL_SHAPE]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x4096xf16>, tensor<4096x1024xf16>) outs(%[[INIT]] : tensor<?x1024xf16>) : tensor<?x1024xf16>
// POPULATE-NEXT: return %[[RESULT]] : tensor<?x1024xf16>
// POPULATE-NEXT: }
func.func @matmul_2d(
    %ctx: !hipsr.context, %a: tensor<?x4096xf16>,
    %b: tensor<4096x1024xf16>) -> tensor<?x1024xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?x4096xf16>, tensor<4096x1024xf16>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x1024xf16>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?x4096xf16>, tensor<4096x1024xf16>)
      outs(%init : tensor<?x1024xf16>) : tensor<?x1024xf16>
  return %result : tensor<?x1024xf16>
}

// -----

// A 1-D left operand is promoted and its leading unit dimension is removed.
// POPULATE-LABEL: func.func @matmul_1d_a(
// POPULATE-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<4096xf16>, %[[B:.+]]: tensor<4096x1024xf16>) -> tensor<1024xf16> {
// POPULATE-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<4096xf16>, tensor<4096x1024xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<1024xf16> shape_region {
// POPULATE-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// POPULATE: %[[ONE_SHAPE:.+]] = shape.from_extents %[[ONE:.+]] : index
// POPULATE-NEXT: %[[A_PROMOTED:.+]] = shape.concat %[[ONE_SHAPE]], %[[A_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: %[[B_PROMOTED:.+]] = shape.concat %[[B_SHAPE]], %[[ONE_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: %[[A_NORMALIZED:.+]] = arith.select %[[A_IS_VECTOR:.+]], %[[A_PROMOTED]], %[[A_SHAPE]] : !shape.shape
// POPULATE-NEXT: %[[B_NORMALIZED:.+]] = arith.select %[[B_IS_VECTOR:.+]], %[[B_PROMOTED]], %[[B_SHAPE]] : !shape.shape
// POPULATE: hipsr.shape_yield %[[RESULT_SHAPE:.+]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<4096xf16>, tensor<4096x1024xf16>) outs(%[[INIT]] : tensor<1024xf16>) : tensor<1024xf16>
// POPULATE-NEXT: return %[[RESULT]] : tensor<1024xf16>
// POPULATE-NEXT: }
func.func @matmul_1d_a(
    %ctx: !hipsr.context, %a: tensor<4096xf16>,
    %b: tensor<4096x1024xf16>) -> tensor<1024xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<4096xf16>, tensor<4096x1024xf16>)
      {type = #hipsr.placeholder_type<normal>} : tensor<1024xf16>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<4096xf16>, tensor<4096x1024xf16>)
      outs(%init : tensor<1024xf16>) : tensor<1024xf16>
  return %result : tensor<1024xf16>
}

// -----

// A 1-D right operand is promoted and its trailing unit dimension is removed.
// POPULATE-LABEL: func.func @matmul_1d_b(
// POPULATE-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<64x4096xf16>, %[[B:.+]]: tensor<4096xf16>) -> tensor<64xf16> {
// POPULATE-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<64x4096xf16>, tensor<4096xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<64xf16> shape_region {
// POPULATE-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// POPULATE: %[[ONE_SHAPE:.+]] = shape.from_extents %[[ONE:.+]] : index
// POPULATE-NEXT: %[[A_PROMOTED:.+]] = shape.concat %[[ONE_SHAPE]], %[[A_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: %[[B_PROMOTED:.+]] = shape.concat %[[B_SHAPE]], %[[ONE_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: %[[A_NORMALIZED:.+]] = arith.select %[[A_IS_VECTOR:.+]], %[[A_PROMOTED]], %[[A_SHAPE]] : !shape.shape
// POPULATE-NEXT: %[[B_NORMALIZED:.+]] = arith.select %[[B_IS_VECTOR:.+]], %[[B_PROMOTED]], %[[B_SHAPE]] : !shape.shape
// POPULATE: hipsr.shape_yield %[[RESULT_SHAPE:.+]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<64x4096xf16>, tensor<4096xf16>) outs(%[[INIT]] : tensor<64xf16>) : tensor<64xf16>
// POPULATE-NEXT: return %[[RESULT]] : tensor<64xf16>
// POPULATE-NEXT: }
func.func @matmul_1d_b(
    %ctx: !hipsr.context, %a: tensor<64x4096xf16>,
    %b: tensor<4096xf16>) -> tensor<64xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<64x4096xf16>, tensor<4096xf16>)
      {type = #hipsr.placeholder_type<normal>} : tensor<64xf16>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<64x4096xf16>, tensor<4096xf16>)
      outs(%init : tensor<64xf16>) : tensor<64xf16>
  return %result : tensor<64xf16>
}

// -----

// A scalar result still yields one !shape.shape value.
// POPULATE-LABEL: func.func @matmul_both_1d(
// POPULATE-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<4096xf16>, %[[B:.+]]: tensor<4096xf16>) -> tensor<f16> {
// POPULATE-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<4096xf16>, tensor<4096xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<f16> shape_region {
// POPULATE-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// POPULATE: %[[EMPTY_SHAPE:.+]] = shape.from_extents  :
// POPULATE: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS:.+]] -> (!shape.shape) {
// POPULATE: %[[RESULT_M:.+]] = arith.select %[[A_IS_VECTOR:.+]], %[[EMPTY_SHAPE]], %[[M:.+]] : !shape.shape
// POPULATE-NEXT: %[[RESULT_N:.+]] = arith.select %[[B_IS_VECTOR:.+]], %[[EMPTY_SHAPE]], %[[N:.+]] : !shape.shape
// POPULATE-NEXT: %[[MATRIX:.+]] = shape.concat %[[RESULT_M]], %[[RESULT_N]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: %[[FINAL_SHAPE:.+]] = shape.concat %[[BATCH:.+]], %[[MATRIX]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: shape.assuming_yield %[[FINAL_SHAPE]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<4096xf16>, tensor<4096xf16>) outs(%[[INIT]] : tensor<f16>) : tensor<f16>
// POPULATE-NEXT: return %[[RESULT]] : tensor<f16>
// POPULATE-NEXT: }
func.func @matmul_both_1d(
    %ctx: !hipsr.context, %a: tensor<4096xf16>, %b: tensor<4096xf16>)
    -> tensor<f16> {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<4096xf16>, tensor<4096xf16>)
      {type = #hipsr.placeholder_type<normal>} : tensor<f16>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<4096xf16>, tensor<4096xf16>)
      outs(%init : tensor<f16>) : tensor<f16>
  return %result : tensor<f16>
}

// -----

// Batch shapes are checked and broadcast before matrix dimensions are joined.
// POPULATE-LABEL: func.func @matmul_batched(
// POPULATE-SAME: %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<?x8x64x4096xf16>, %[[B:.+]]: tensor<1x8x4096x1024xf16>) -> tensor<?x8x64x1024xf16> {
// POPULATE-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x8x64x4096xf16>, tensor<1x8x4096x1024xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<?x8x64x1024xf16> shape_region {
// POPULATE-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// POPULATE: %[[BATCH_WITNESS:.+]] = shape.cstr_broadcastable %[[A_BATCH:.+]], %[[B_BATCH:.+]] : !shape.shape, !shape.shape
// POPULATE-NEXT: %[[WITNESS:.+]] = shape.assuming_all %[[K_WITNESS:.+]], %[[BATCH_WITNESS]]
// POPULATE-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// POPULATE-NEXT: %[[BATCH:.+]] = shape.broadcast %[[A_BATCH]], %[[B_BATCH]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE: %[[MATRIX:.+]] = shape.concat %[[RESULT_M:.+]], %[[RESULT_N:.+]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: %[[FINAL_SHAPE:.+]] = shape.concat %[[BATCH]], %[[MATRIX]] : !shape.shape, !shape.shape -> !shape.shape
// POPULATE-NEXT: shape.assuming_yield %[[FINAL_SHAPE]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: %[[RESULT:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x8x64x4096xf16>, tensor<1x8x4096x1024xf16>) outs(%[[INIT]] : tensor<?x8x64x1024xf16>) : tensor<?x8x64x1024xf16>
// POPULATE-NEXT: return %[[RESULT]] : tensor<?x8x64x1024xf16>
// POPULATE-NEXT: }
func.func @matmul_batched(
    %ctx: !hipsr.context, %a: tensor<?x8x64x4096xf16>,
    %b: tensor<1x8x4096x1024xf16>) -> tensor<?x8x64x1024xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%a, %b : tensor<?x8x64x4096xf16>,
                       tensor<1x8x4096x1024xf16>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x8x64x1024xf16>
  %result = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<?x8x64x4096xf16>,
                    tensor<1x8x4096x1024xf16>)
      outs(%init : tensor<?x8x64x1024xf16>) : tensor<?x8x64x1024xf16>
  return %result : tensor<?x8x64x1024xf16>
}
