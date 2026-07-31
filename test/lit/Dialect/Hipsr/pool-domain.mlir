// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// REQUIRES: hipsr
//
//===----------------------------------------------------------------------===//
// Checks pool-domain syntax, value forwarding, and invalid IR diagnostics.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// The operand is forwarded to the block argument, and the two yielded values
// become two results.
// CHECK-LABEL: func.func @roundtrip(
// CHECK-SAME: %[[INPUT:.+]]: tensor<3x4xf32>) -> (tensor<2x?xi64>, tensor<i32>) {
// CHECK-NEXT: %[[RESULTS:.+]]:2 = hipsr.pool_domain(%[[INPUT]] : tensor<3x4xf32>) {
// CHECK-NEXT: ^bb0(%[[DOMAIN_INPUT:.+]]: tensor<3x4xf32>):
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[DIM:.+]] = tensor.dim %[[DOMAIN_INPUT]], %[[C1]] : tensor<3x4xf32>
// CHECK-NEXT: %[[INDEX_BUFFER:.+]] = tensor.empty(%[[DIM]]) : tensor<2x?xi64>
// CHECK-NEXT: %[[COUNT_BUFFER:.+]] = tensor.empty() : tensor<i32>
// CHECK-NEXT: hipsr.pool_domain_yield %[[INDEX_BUFFER]], %[[COUNT_BUFFER]] : tensor<2x?xi64>, tensor<i32>
// CHECK-NEXT: } -> tensor<2x?xi64>, tensor<i32>
// CHECK-NEXT: return %[[RESULTS]]#0, %[[RESULTS]]#1 : tensor<2x?xi64>, tensor<i32>
// CHECK-NEXT: }
func.func @roundtrip(%in: tensor<3x4xf32>) -> (tensor<2x?xi64>, tensor<i32>) {
  %0:2 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
  ^bb0(%domain_in: tensor<3x4xf32>):
    %c1 = arith.constant 1 : index
    %n = tensor.dim %domain_in, %c1 : tensor<3x4xf32>
    %idx = tensor.empty(%n) : tensor<2x?xi64>
    %cnt = tensor.empty() : tensor<i32>
    hipsr.pool_domain_yield %idx, %cnt : tensor<2x?xi64>, tensor<i32>
  } -> tensor<2x?xi64>, tensor<i32>
  return %0#0, %0#1 : tensor<2x?xi64>, tensor<i32>
}

// -----
// A hand-written post-bufferization form accepts memref and index values.
// CHECK-LABEL: func.func @memref_form(
// CHECK-SAME: %[[INPUT:.+]]: memref<2x?xi64>) -> (memref<2x?xi64>, index) {
// CHECK-NEXT: %[[RESULTS:.+]]:2 = hipsr.pool_domain(%[[INPUT]] : memref<2x?xi64>) {
// CHECK-NEXT: ^bb0(%[[DOMAIN_INPUT:.+]]: memref<2x?xi64>):
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[DIM:.+]] = memref.dim %[[DOMAIN_INPUT]], %[[C1]] : memref<2x?xi64>
// CHECK-NEXT: hipsr.pool_domain_yield %[[DOMAIN_INPUT]], %[[DIM]] : memref<2x?xi64>, index
// CHECK-NEXT: } -> memref<2x?xi64>, index
// CHECK-NEXT: return %[[RESULTS]]#0, %[[RESULTS]]#1 : memref<2x?xi64>, index
// CHECK-NEXT: }
func.func @memref_form(%in: memref<2x?xi64>) -> (memref<2x?xi64>, index) {
  %0:2 = hipsr.pool_domain(%in : memref<2x?xi64>) {
  ^bb0(%domain_in: memref<2x?xi64>):
    %c1 = arith.constant 1 : index
    %n = memref.dim %domain_in, %c1 : memref<2x?xi64>
    hipsr.pool_domain_yield %domain_in, %n : memref<2x?xi64>, index
  } -> memref<2x?xi64>, index
  return %0#0, %0#1 : memref<2x?xi64>, index
}

// -----
// The printer omits the empty implicit yield.
// CHECK-LABEL: func.func @empty_domain() {
// CHECK-NEXT: hipsr.pool_domain() {
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @empty_domain() {
  hipsr.pool_domain() {
    hipsr.pool_domain_yield
  }
  return
}

// -----
// The generic form is invalid because the body has no block.
func.func @empty_body() {
  // expected-error @+1 {{failed to verify constraint: region with 1 blocks}}
  "hipsr.pool_domain"() ({
  }) : () -> ()
  return
}

// -----
// The op has one operand, but the body has no entry block argument.
func.func @missing_entry_argument(%in: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // expected-error @+2 {{along control flow edge from parent to Region #0: region branch point has 1 operands, but region successor needs 0 inputs}}
  // expected-note @+1 {{region branch point}}
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
    %local = tensor.empty() : tensor<3x4xf32>
    hipsr.pool_domain_yield %local : tensor<3x4xf32>
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
// The body has one entry block argument, but the op has no operand.
func.func @extra_entry_argument() -> tensor<3x4xf32> {
  // expected-error @+2 {{along control flow edge from parent to Region #0: region branch point has 0 operands, but region successor needs 1 inputs}}
  // expected-note @+1 {{region branch point}}
  %0 = hipsr.pool_domain() {
  ^bb0(%domain_in: tensor<3x4xf32>):
    hipsr.pool_domain_yield %domain_in : tensor<3x4xf32>
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
// The f32 pool-domain operand does not match the i64 entry block argument.
func.func @entry_argument_type_mismatch(%in: tensor<3x4xf32>)
    -> tensor<3x4xi64> {
  // expected-error @+2 {{along control flow edge from parent to Region #0: successor operand type #0 'tensor<3x4xf32>' should match successor input type #0 'tensor<3x4xi64>'}}
  // expected-note @+1 {{region branch point}}
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
  ^bb0(%domain_in: tensor<3x4xi64>):
    hipsr.pool_domain_yield %domain_in : tensor<3x4xi64>
  } -> tensor<3x4xi64>
  return %0 : tensor<3x4xi64>
}

// -----
// The pool domain has one result, but the yield has no value.
func.func @missing_yield_value() -> tensor<3x4xf32> {
  // expected-error @+1 {{along control flow edge from Operation hipsr.pool_domain_yield to parent: region branch point has 0 operands, but region successor needs 1 inputs}}
  %0 = hipsr.pool_domain() {
    // expected-note @+1 {{region branch point}}
    hipsr.pool_domain_yield
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
// The yield has one value, but the pool domain has no result.
func.func @extra_yield_value() {
  // expected-error @+1 {{along control flow edge from Operation hipsr.pool_domain_yield to parent: region branch point has 1 operands, but region successor needs 0 inputs}}
  hipsr.pool_domain() {
    %local = tensor.empty() : tensor<3x4xf32>
    // expected-note @+1 {{region branch point}}
    hipsr.pool_domain_yield %local : tensor<3x4xf32>
  }
  return
}

// -----
// The yielded f32 value does not match the pool domain's i64 result.
func.func @yield_type_mismatch() -> tensor<3x4xi64> {
  // expected-error @+1 {{along control flow edge from Operation hipsr.pool_domain_yield to parent: successor operand type #0 'tensor<3x4xf32>' should match successor input type #0 'tensor<3x4xi64>'}}
  %0 = hipsr.pool_domain() {
    %local = tensor.empty() : tensor<3x4xf32>
    // expected-note @+1 {{region branch point}}
    hipsr.pool_domain_yield %local : tensor<3x4xf32>
  } -> tensor<3x4xi64>
  return %0 : tensor<3x4xi64>
}

// -----
// The isolated body uses the parent value instead of its block argument.
func.func @body_uses_parent_value_directly(%in: tensor<3x4xf32>)
    -> tensor<3x4xf32> {
  // expected-note @+1 {{required by region isolation constraints}}
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
  ^bb0(%domain_in: tensor<3x4xf32>):
    // expected-error @+1 {{using value defined outside the region}}
    hipsr.pool_domain_yield %in : tensor<3x4xf32>
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
// The yield is invalid because it has no parent pool domain.
func.func @yield_without_parent(%in: tensor<3x4xf32>) {
  // expected-error @+1 {{expects parent op 'hipsr.pool_domain'}}
  hipsr.pool_domain_yield %in : tensor<3x4xf32>
}

// -----
// The body has two blocks, but a pool domain allows only one.
func.func @multi_block_body() {
  // expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
  hipsr.pool_domain() {
    hipsr.pool_domain_yield
  ^bb1:
    hipsr.pool_domain_yield
  }
  return
}
