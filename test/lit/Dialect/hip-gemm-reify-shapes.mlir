// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// CHECK-LABEL: func.func @gemm_dynamic_default
// CHECK-SAME: (%{{.*}}: !hip.context, %[[A:[A-Za-z0-9_]+]]: tensor<?x4xf16>, %[[B:[A-Za-z0-9_]+]]: tensor<4x?xf16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[M:.*]] = tensor.dim %[[A]], %[[C0]]
// CHECK-DAG: %[[N:.*]] = tensor.dim %[[B]], %[[C1]]
// CHECK: return %[[M]], %[[N]]
func.func @gemm_dynamic_default(
    %ctx: !hip.context, %a: tensor<?x4xf16>, %b: tensor<4x?xf16>,
    %out: tensor<?x?xf16>) -> (index, index) {
  %r = hip.gemm(%ctx)
    ins(%a, %b : tensor<?x4xf16>, tensor<4x?xf16>)
    outs(%out : tensor<?x?xf16>)
    {transA = 0 : i64, transB = 0 : i64} : tensor<?x?xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %r, %c1 : tensor<?x?xf16>
  return %d0, %d1 : index, index
}

// Both transposed: M=A.dim1, N=B.dim0. Optional C does not supply extents.
// CHECK-LABEL: func.func @gemm_dynamic_transposed
// CHECK-SAME: (%{{.*}}: !hip.context, %[[A:[A-Za-z0-9_]+]]: tensor<4x?xf16>, %[[B:[A-Za-z0-9_]+]]: tensor<?x4xf16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[M:.*]] = tensor.dim %[[A]], %[[C1]]
// CHECK-DAG: %[[N:.*]] = tensor.dim %[[B]], %[[C0]]
// CHECK: return %[[M]], %[[N]]
func.func @gemm_dynamic_transposed(
    %ctx: !hip.context, %a: tensor<4x?xf16>, %b: tensor<?x4xf16>,
    %c: tensor<?xf16>, %out: tensor<?x?xf16>) -> (index, index) {
  %r = hip.gemm(%ctx)
    ins(%a, %b, %c : tensor<4x?xf16>, tensor<?x4xf16>, tensor<?xf16>)
    outs(%out : tensor<?x?xf16>)
    {transA = 1 : i64, transB = 1 : i64} : tensor<?x?xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %r, %c1 : tensor<?x?xf16>
  return %d0, %d1 : index, index
}
