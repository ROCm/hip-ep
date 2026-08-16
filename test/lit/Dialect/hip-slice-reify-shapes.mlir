// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// The exact extent operands, not the destination capacity or raw parameter
// tensors, are the sole reification authority.
// CHECK-LABEL: func.func @exact_extents
// CHECK-SAME: %[[E0:[^,]+]]: index
// CHECK-SAME: %[[E1:[^)]+]]: index
// CHECK-NOT: tensor.dim
// CHECK: return %[[E0]], %[[E1]] : index, index
func.func @exact_extents(
    %ctx: !hip.context, %data: tensor<?x?xf32>, %valid: i1,
    %s0: i64, %s1: i64, %p0: i64, %p1: i64,
    %e0: index, %e1: index) -> (index, index) {
  %init = tensor.empty(%e0, %e1) : tensor<?x?xf32>
  %result = hip.slice(%ctx) ins(%data : tensor<?x?xf32>)
      valid(%valid)
      starts(%s0, %s1 : i64, i64)
      steps(%p0, %p1 : i64, i64)
      extents(%e0, %e1 : index, index)
      outs(%init : tensor<?x?xf32>)
      : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}
