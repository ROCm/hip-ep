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
// POPULATE-NEXT: %[[INIT:.+]] = hipsr.placeholder(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x4096xf16>, tensor<4096x1024xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<?x1024xf16> shape_region {
// POPULATE-NEXT: ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[B_SHAPE:.+]]: !shape.shape):
// POPULATE: %[[K_WITNESS:.+]] = shape.cstr_eq
// POPULATE-NEXT: %[[BATCH_WITNESS:.+]] = shape.cstr_broadcastable
// POPULATE-NEXT: %[[WITNESS:.+]] = shape.assuming_all %[[K_WITNESS]], %[[BATCH_WITNESS]]
// POPULATE-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// POPULATE: shape.assuming_yield
// POPULATE-NEXT: }
// POPULATE-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: %[[RESULT:.+]] = hipsr.matmul
// POPULATE-NOT: shape_region
// POPULATE-NEXT: return %[[RESULT]] : tensor<?x1024xf16>
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
// POPULATE-LABEL: func.func @matmul_1d_a
// POPULATE: hipsr.placeholder
// POPULATE: shape.concat
// POPULATE: arith.select
// POPULATE: hipsr.shape_yield %{{.+}} : !shape.shape
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
// POPULATE-LABEL: func.func @matmul_1d_b
// POPULATE: hipsr.placeholder
// POPULATE: shape.concat
// POPULATE: arith.select
// POPULATE: hipsr.shape_yield %{{.+}} : !shape.shape
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
// POPULATE-LABEL: func.func @matmul_both_1d
// POPULATE: hipsr.placeholder
// POPULATE-SAME: tensor<f16> shape_region {
// POPULATE: hipsr.shape_yield %{{.+}} : !shape.shape
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
// POPULATE-LABEL: func.func @matmul_batched
// POPULATE: shape.cstr_broadcastable
// POPULATE: shape.broadcast
// POPULATE: hipsr.shape_yield %{{.+}} : !shape.shape
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
