// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// -----
// CHECK-LABEL: func.func @roundtrip
// CHECK: %[[R:.*]]:2 = hipsr.pool_domain(%{{.*}} : tensor<3x4xf32>) {
// CHECK-NEXT: ^bb0(%[[IN:.*]]: tensor<3x4xf32>):
// CHECK:   %[[N:.*]] = tensor.dim %[[IN]]
// CHECK:   %[[IDX:.*]] = tensor.empty(%[[N]]) : tensor<2x?xi64>
// CHECK:   %[[CNT:.*]] = tensor.empty() : tensor<i32>
// CHECK:   hipsr.pool_domain_yield %[[IDX]], %[[CNT]] : tensor<2x?xi64>, tensor<i32>
// CHECK: } -> tensor<2x?xi64>, tensor<i32>

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
// CHECK-LABEL: func.func @memref_form
// CHECK: hipsr.pool_domain(%{{.*}} : memref<2x?xi64>)
// CHECK-NEXT: ^bb0(%[[IN:.*]]: memref<2x?xi64>):
// CHECK: hipsr.pool_domain_yield %[[IN]], %{{.*}} : memref<2x?xi64>, index
// CHECK: -> memref<2x?xi64>, index

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
// CHECK-LABEL: func.func @empty_domain
// CHECK: hipsr.pool_domain() {
// CHECK-NEXT: }

func.func @empty_domain() {
  hipsr.pool_domain() {
    hipsr.pool_domain_yield
  }
  return
}

// -----
func.func @empty_body() {
  // expected-error @+1 {{failed to verify constraint: region with 1 blocks}}
  "hipsr.pool_domain"() ({
  }) : () -> ()
  return
}

// -----
func.func @missing_entry_argument(%in: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // expected-error @+2 {{region branch point has 1 operands, but region successor needs 0 inputs}}
  // expected-note @+1 {{region branch point}}
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
    %local = tensor.empty() : tensor<3x4xf32>
    hipsr.pool_domain_yield %local : tensor<3x4xf32>
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
func.func @extra_entry_argument() -> tensor<3x4xf32> {
  // expected-error @+2 {{region branch point has 0 operands, but region successor needs 1 inputs}}
  // expected-note @+1 {{region branch point}}
  %0 = hipsr.pool_domain() {
  ^bb0(%domain_in: tensor<3x4xf32>):
    hipsr.pool_domain_yield %domain_in : tensor<3x4xf32>
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
func.func @entry_argument_type_mismatch(%in: tensor<3x4xf32>)
    -> tensor<3x4xi64> {
  // expected-error @+2 {{successor operand type #0 'tensor<3x4xf32>' should match successor input type #0 'tensor<3x4xi64>'}}
  // expected-note @+1 {{region branch point}}
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
  ^bb0(%domain_in: tensor<3x4xi64>):
    hipsr.pool_domain_yield %domain_in : tensor<3x4xi64>
  } -> tensor<3x4xi64>
  return %0 : tensor<3x4xi64>
}

// -----
func.func @missing_yield_value() -> tensor<3x4xf32> {
  // expected-error @+1 {{region branch point has 0 operands, but region successor needs 1 inputs}}
  %0 = hipsr.pool_domain() {
    // expected-note @+1 {{region branch point}}
    hipsr.pool_domain_yield
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
func.func @extra_yield_value() {
  // expected-error @+1 {{region branch point has 1 operands, but region successor needs 0 inputs}}
  hipsr.pool_domain() {
    %local = tensor.empty() : tensor<3x4xf32>
    // expected-note @+1 {{region branch point}}
    hipsr.pool_domain_yield %local : tensor<3x4xf32>
  }
  return
}

// -----
func.func @yield_type_mismatch() -> tensor<3x4xi64> {
  // expected-error @+1 {{successor operand type #0 'tensor<3x4xf32>' should match successor input type #0 'tensor<3x4xi64>'}}
  %0 = hipsr.pool_domain() {
    %local = tensor.empty() : tensor<3x4xf32>
    // expected-note @+1 {{region branch point}}
    hipsr.pool_domain_yield %local : tensor<3x4xf32>
  } -> tensor<3x4xi64>
  return %0 : tensor<3x4xi64>
}

// -----
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
func.func @yield_without_parent(%in: tensor<3x4xf32>) {
  // expected-error @+1 {{expects parent op 'hipsr.pool_domain'}}
  hipsr.pool_domain_yield %in : tensor<3x4xf32>
}

// -----
func.func @multi_block_body() {
  // expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
  hipsr.pool_domain() {
    hipsr.pool_domain_yield
  ^bb1:
    hipsr.pool_domain_yield
  }
  return
}
