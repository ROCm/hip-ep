// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// CSE and canonicalize must preserve the IsolatedFromAbove boundary.

// RUN: hip-mlir-opt -cse -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @standard_pass_boundary
// CHECK: %[[OUTSIDE_C2:.*]] = arith.constant 2 : i32
// CHECK: %[[DOMAIN:.*]] = hipsr.pool_domain(
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: i32):
// CHECK: %[[INSIDE_C2:.*]] = arith.constant 2 : i32
// CHECK: %[[SCALED:.*]] = arith.muli %[[ARG]], %[[INSIDE_C2]] : i32
// CHECK: hipsr.pool_domain_yield %[[SCALED]] : i32
// CHECK: %[[RESULT:.*]] = arith.addi %[[DOMAIN]], %[[OUTSIDE_C2]] : i32
// CHECK: return %[[RESULT]] : i32

func.func @standard_pass_boundary(%arg: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %scaled = hipsr.pool_domain(%arg : i32) {
  ^bb0(%domain_arg: i32):
    %inner_c2 = arith.constant 2 : i32
    %inner_scaled = arith.muli %domain_arg, %inner_c2 : i32
    hipsr.pool_domain_yield %inner_scaled : i32
  } -> i32
  %result = arith.addi %scaled, %c2 : i32
  return %result : i32
}
