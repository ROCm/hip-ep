// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file -cse %s | FileCheck %s --check-prefix=CSE
// RUN: hip-mlir-opt --split-input-file -canonicalize %s | FileCheck %s --check-prefix=CANON

// Issue #512, case 1: the same constant appears on both sides of the
// IsolatedFromAbove boundary. CSE must keep the inner constant in the domain.
// CSE-LABEL: func.func @cse_keeps_domain_isolated(
// CSE-SAME: %[[INPUT:.+]]: i32) -> i32 {
// CSE-NEXT: %[[OUTSIDE_C2:.+]] = arith.constant 2 : i32
// CSE-NEXT: %[[DOMAIN:.+]] = hipsr.pool_domain(%[[INPUT]] : i32) {
// CSE-NEXT: ^bb0(%[[DOMAIN_INPUT:.+]]: i32):
// CSE-NEXT: %[[INSIDE_C2:.+]] = arith.constant 2 : i32
// CSE-NEXT: %[[SCALED:.+]] = arith.muli %[[DOMAIN_INPUT]], %[[INSIDE_C2]] : i32
// CSE-NEXT: hipsr.pool_domain_yield %[[SCALED]] : i32
// CSE-NEXT: } -> i32
// CSE-NEXT: %[[RESULT:.+]] = arith.addi %[[DOMAIN]], %[[OUTSIDE_C2]] : i32
// CSE-NEXT: return %[[RESULT]] : i32
// CSE-NEXT: }
func.func @cse_keeps_domain_isolated(%arg: i32) -> i32 {
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

// -----
// Issue #512, case 2: canonicalize folds the addition inside the
// IsolatedFromAbove boundary. The new constant must stay in the domain.
// CANON-LABEL: func.func @canonicalize_keeps_domain_isolated(
// CANON-SAME: %[[INPUT:.+]]: i32) -> i32 {
// CANON-NEXT: %[[OUTSIDE_C2:.+]] = arith.constant 2 : i32
// CANON-NEXT: %[[DOMAIN:.+]] = hipsr.pool_domain(%[[INPUT]] : i32) {
// CANON-NEXT: ^bb0(%[[DOMAIN_INPUT:.+]]: i32):
// CANON-NEXT: %[[INSIDE_C2:.+]] = arith.constant 2 : i32
// CANON-NEXT: %[[SCALED:.+]] = arith.muli %[[DOMAIN_INPUT]], %[[INSIDE_C2]] : i32
// CANON-NEXT: hipsr.pool_domain_yield %[[SCALED]] : i32
// CANON-NEXT: } -> i32
// CANON-NEXT: %[[RESULT:.+]] = arith.addi %[[DOMAIN]], %[[OUTSIDE_C2]] : i32
// CANON-NEXT: return %[[RESULT]] : i32
// CANON-NEXT: }
func.func @canonicalize_keeps_domain_isolated(%arg: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %scaled = hipsr.pool_domain(%arg : i32) {
  ^bb0(%domain_arg: i32):
    %c1 = arith.constant 1 : i32
    %inner_c2 = arith.addi %c1, %c1 : i32
    %inner_scaled = arith.muli %domain_arg, %inner_c2 : i32
    hipsr.pool_domain_yield %inner_scaled : i32
  } -> i32
  %result = arith.addi %scaled, %c2 : i32
  return %result : i32
}
