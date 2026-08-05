// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Positive coverage for -hipsr-materialize-init-tensors.
//
// This file grows with the pass instead of gaining a sibling per phase: each
// case keeps its input IR and only its CHECK lines are extended as later phases
// land. What is checked today is phase 1, placeholder grouping -- every
// hipsr.placeholder in a domain moves to the front of the domain block, keeping
// the relative order SSA form already gives it. Once the shape regions become
// scf.execute_region and the placeholders become tensor.empty, the same cases
// check that instead.
//
// Error paths live in invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -hipsr-materialize-init-tensors | FileCheck %s

// The canonical single-placeholder domain, and the case that carries the most
// CHECK detail as the later phases land. Its placeholder already leads the
// block, so phase 1 leaves the domain as it is.
// CHECK-LABEL: func.func @matmul_domain(
// CHECK:         ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DA:.+]]: tensor<?x256xf16>, %[[DB:.+]]: tensor<256x512xf16>):
// CHECK:           %[[INIT:.+]] = hipsr.placeholder(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<?x256xf16>, tensor<256x512xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x512xf16> shape_region {
// CHECK:             hipsr.shape_yield2 %{{.+}} : !shape.shape
// CHECK:           %[[MATMUL:.+]] = hipsr.matmul(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<?x256xf16>, tensor<256x512xf16>) outs(%[[INIT]] : tensor<?x512xf16>) : tensor<?x512xf16>
// CHECK:           hipsr.pool_domain_yield %[[MATMUL]] : tensor<?x512xf16>
func.func @matmul_domain(%ctx: !hipsr.context, %a: tensor<?x256xf16>,
                         %b: tensor<256x512xf16>) -> tensor<?x512xf16> {
  %0 = hipsr.pool_domain(%ctx, %a, %b
      : !hipsr.context, tensor<?x256xf16>, tensor<256x512xf16>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x256xf16>,
       %domain_b: tensor<256x512xf16>):
    %init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16> shape_region {
    ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
      %m_index = shape.const_size 0
      %m = shape.get_extent %a_shape, %m_index
          : !shape.shape, !shape.size -> !shape.size
      %n_index = shape.const_size 1
      %n = shape.get_extent %b_shape, %n_index
          : !shape.shape, !shape.size -> !shape.size
      %result_shape = shape.from_extents %m, %n : !shape.size, !shape.size
      hipsr.shape_yield2 %result_shape : !shape.shape
    }
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        outs(%init : tensor<?x512xf16>) : tensor<?x512xf16>
    hipsr.pool_domain_yield %matmul : tensor<?x512xf16>
  } -> tensor<?x512xf16>
  return %0 : tensor<?x512xf16>
}

// -----

// The add placeholder is defined after the matmul in the input; grouping pulls
// it in front of both data ops without overtaking the matmul placeholder it
// depends on.
// CHECK-LABEL: func.func @interleaved(
// CHECK:         ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DA:.+]]: tensor<?x256xf16>, %[[DB:.+]]: tensor<256x512xf16>, %[[DC:.+]]: tensor<?x512xf16>):
// CHECK:           %[[MATMUL_INIT:.+]] = hipsr.placeholder(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<?x256xf16>, tensor<256x512xf16>)
// CHECK:           %[[ADD_INIT:.+]] = hipsr.placeholder(%[[DCTX]]) ins(%[[MATMUL_INIT]], %[[DC]] : tensor<?x512xf16>, tensor<?x512xf16>)
// CHECK:           %[[MATMUL:.+]] = hipsr.matmul(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<?x256xf16>, tensor<256x512xf16>) outs(%[[MATMUL_INIT]] : tensor<?x512xf16>) : tensor<?x512xf16>
// CHECK:           %[[ADD:.+]] = hipsr.add(%[[DCTX]]) ins(%[[MATMUL]], %[[DC]] : tensor<?x512xf16>, tensor<?x512xf16>) outs(%[[ADD_INIT]] : tensor<?x512xf16>) : tensor<?x512xf16>
// CHECK:           hipsr.pool_domain_yield %[[ADD]] : tensor<?x512xf16>
func.func @interleaved(%ctx: !hipsr.context, %a: tensor<?x256xf16>,
                       %b: tensor<256x512xf16>, %c: tensor<?x512xf16>)
    -> tensor<?x512xf16> {
  %0 = hipsr.pool_domain(%ctx, %a, %b, %c
      : !hipsr.context, tensor<?x256xf16>, tensor<256x512xf16>,
        tensor<?x512xf16>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x256xf16>,
       %domain_b: tensor<256x512xf16>, %domain_c: tensor<?x512xf16>):
    %matmul_init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16> shape_region {
    ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
      %m_index = shape.const_size 0
      %m = shape.get_extent %a_shape, %m_index
          : !shape.shape, !shape.size -> !shape.size
      %n_index = shape.const_size 1
      %n = shape.get_extent %b_shape, %n_index
          : !shape.shape, !shape.size -> !shape.size
      %matmul_shape = shape.from_extents %m, %n : !shape.size, !shape.size
      hipsr.shape_yield2 %matmul_shape : !shape.shape
    }
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        outs(%matmul_init : tensor<?x512xf16>) : tensor<?x512xf16>
    %add_init = hipsr.placeholder(%domain_ctx)
        ins(%matmul_init, %domain_c : tensor<?x512xf16>, tensor<?x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16> shape_region {
    ^bb0(%lhs_shape: !shape.shape, %rhs_shape: !shape.shape):
      %add_shape = shape.broadcast %lhs_shape, %rhs_shape
          : !shape.shape, !shape.shape -> !shape.shape
      hipsr.shape_yield2 %add_shape : !shape.shape
    }
    %add = hipsr.add(%domain_ctx)
        ins(%matmul, %domain_c : tensor<?x512xf16>, tensor<?x512xf16>)
        outs(%add_init : tensor<?x512xf16>) : tensor<?x512xf16>
    hipsr.pool_domain_yield %add : tensor<?x512xf16>
  } -> tensor<?x512xf16>
  return %0 : tensor<?x512xf16>
}

// -----

// Same domain as @interleaved but already grouped, so phase 1 is a no-op. Two
// leading placeholders in a row pin the skip that keeps a placeholder from
// being moved before itself, which would otherwise reverse their order.
// CHECK-LABEL: func.func @already_grouped(
// CHECK:         ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DA:.+]]: tensor<?x256xf16>, %[[DB:.+]]: tensor<256x512xf16>, %[[DC:.+]]: tensor<?x512xf16>):
// CHECK:           %[[MATMUL_INIT:.+]] = hipsr.placeholder(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<?x256xf16>, tensor<256x512xf16>)
// CHECK:           %[[ADD_INIT:.+]] = hipsr.placeholder(%[[DCTX]]) ins(%[[MATMUL_INIT]], %[[DC]] : tensor<?x512xf16>, tensor<?x512xf16>)
// CHECK:           %[[MATMUL:.+]] = hipsr.matmul(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<?x256xf16>, tensor<256x512xf16>) outs(%[[MATMUL_INIT]] : tensor<?x512xf16>) : tensor<?x512xf16>
// CHECK:           %[[ADD:.+]] = hipsr.add(%[[DCTX]]) ins(%[[MATMUL]], %[[DC]] : tensor<?x512xf16>, tensor<?x512xf16>) outs(%[[ADD_INIT]] : tensor<?x512xf16>) : tensor<?x512xf16>
// CHECK:           hipsr.pool_domain_yield %[[ADD]] : tensor<?x512xf16>
func.func @already_grouped(%ctx: !hipsr.context, %a: tensor<?x256xf16>,
                           %b: tensor<256x512xf16>, %c: tensor<?x512xf16>)
    -> tensor<?x512xf16> {
  %0 = hipsr.pool_domain(%ctx, %a, %b, %c
      : !hipsr.context, tensor<?x256xf16>, tensor<256x512xf16>,
        tensor<?x512xf16>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x256xf16>,
       %domain_b: tensor<256x512xf16>, %domain_c: tensor<?x512xf16>):
    %matmul_init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16> shape_region {
    ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
      %m_index = shape.const_size 0
      %m = shape.get_extent %a_shape, %m_index
          : !shape.shape, !shape.size -> !shape.size
      %n_index = shape.const_size 1
      %n = shape.get_extent %b_shape, %n_index
          : !shape.shape, !shape.size -> !shape.size
      %matmul_shape = shape.from_extents %m, %n : !shape.size, !shape.size
      hipsr.shape_yield2 %matmul_shape : !shape.shape
    }
    %add_init = hipsr.placeholder(%domain_ctx)
        ins(%matmul_init, %domain_c : tensor<?x512xf16>, tensor<?x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16> shape_region {
    ^bb0(%lhs_shape: !shape.shape, %rhs_shape: !shape.shape):
      %add_shape = shape.broadcast %lhs_shape, %rhs_shape
          : !shape.shape, !shape.shape -> !shape.shape
      hipsr.shape_yield2 %add_shape : !shape.shape
    }
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        outs(%matmul_init : tensor<?x512xf16>) : tensor<?x512xf16>
    %add = hipsr.add(%domain_ctx)
        ins(%matmul, %domain_c : tensor<?x512xf16>, tensor<?x512xf16>)
        outs(%add_init : tensor<?x512xf16>) : tensor<?x512xf16>
    hipsr.pool_domain_yield %add : tensor<?x512xf16>
  } -> tensor<?x512xf16>
  return %0 : tensor<?x512xf16>
}

// -----

// A domain without placeholders is left alone.
// CHECK-LABEL: func.func @no_placeholder(
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         %[[DIM:.+]] = tensor.dim %{{.+}}, %[[C1]] : tensor<3x4xf32>
// CHECK:         %[[BUFFER:.+]] = tensor.empty(%[[DIM]]) : tensor<2x?xi64>
// CHECK:         hipsr.pool_domain_yield %[[BUFFER]] : tensor<2x?xi64>
func.func @no_placeholder(%in: tensor<3x4xf32>) -> tensor<2x?xi64> {
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
  ^bb0(%domain_in: tensor<3x4xf32>):
    %c1 = arith.constant 1 : index
    %n = tensor.dim %domain_in, %c1 : tensor<3x4xf32>
    %buffer = tensor.empty(%n) : tensor<2x?xi64>
    hipsr.pool_domain_yield %buffer : tensor<2x?xi64>
  } -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}
