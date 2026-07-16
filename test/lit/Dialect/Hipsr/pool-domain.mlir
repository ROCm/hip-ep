// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Tests that hipsr.pool_domain / hipsr.pool_domain_yield parse, round-trip, and
// that the verifiers reject malformed IR.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// -----
// Round-trip: operands + results. The body reads an operand (allowed by
// IsolatedFromAboveButAllowOperands) to compute a dynamic shape.
// CHECK-LABEL: func.func @roundtrip
// CHECK: %[[R:.*]]:2 = hipsr.pool_domain(%{{.*}} : tensor<3x4xf32>) {
// CHECK:   %[[N:.*]] = tensor.dim
// CHECK:   %[[IDX:.*]] = tensor.empty(%[[N]]) : tensor<2x?xi64>
// CHECK:   %[[CNT:.*]] = tensor.empty() : tensor<i32>
// CHECK:   hipsr.pool_domain_yield %[[IDX]], %[[CNT]] : tensor<2x?xi64>, tensor<i32>
// CHECK: } -> tensor<2x?xi64>, tensor<i32>
func.func @roundtrip(%in: tensor<3x4xf32>) -> (tensor<2x?xi64>, tensor<i32>) {
  %0:2 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
    %c1 = arith.constant 1 : index
    %n = tensor.dim %in, %c1 : tensor<3x4xf32>
    %idx = tensor.empty(%n) : tensor<2x?xi64>
    %cnt = tensor.empty() : tensor<i32>
    hipsr.pool_domain_yield %idx, %cnt : tensor<2x?xi64>, tensor<i32>
  } -> tensor<2x?xi64>, tensor<i32>
  return %0#0, %0#1 : tensor<2x?xi64>, tensor<i32>
}

// -----
// Post-bufferization form: memref operands/results (AnyType) plus a non-tensor
// `index` result.
// CHECK-LABEL: func.func @memref_form
// CHECK: hipsr.pool_domain(%{{.*}} : memref<2x?xi64>)
// CHECK: hipsr.pool_domain_yield %{{.*}}, %{{.*}} : memref<2x?xi64>, index
// CHECK: -> memref<2x?xi64>, index
func.func @memref_form(%in: memref<2x?xi64>) -> (memref<2x?xi64>, index) {
  %0:2 = hipsr.pool_domain(%in : memref<2x?xi64>) {
    %c1 = arith.constant 1 : index
    %n = memref.dim %in, %c1 : memref<2x?xi64>
    hipsr.pool_domain_yield %in, %n : memref<2x?xi64>, index
  } -> memref<2x?xi64>, index
  return %0#0, %0#1 : memref<2x?xi64>, index
}

// -----
// No operands, no results: the empty implicit terminator has no operands, so
// the assembly format elides it -- the region round-trips as just `{ }`.
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
// Verifier: result count must match the yielded value count.
func.func @result_count_mismatch(%in: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // expected-error @+1 {{has 1 result(s) but its pool_domain_yield yields 0 value(s)}}
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
    hipsr.pool_domain_yield
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
// Verifier: result type must match the yielded value type.
func.func @result_type_mismatch(%in: tensor<3x4xf32>) -> tensor<3x4xi64> {
  // expected-error @+1 {{result #0 type 'tensor<3x4xi64>' does not match the yielded value type 'tensor<3x4xf32>'}}
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
    hipsr.pool_domain_yield %in : tensor<3x4xf32>
  } -> tensor<3x4xi64>
  return %0 : tensor<3x4xi64>
}

// -----
// Verifier (IsolatedFromAboveButAllowOperands): the body may not use a value
// from the enclosing scope that is not one of the op's operands.
func.func @body_uses_outside_value(%in: tensor<3x4xf32>, %other: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // Error is on the inner op using the outside value; note on the pool_domain.
  // expected-note @+1 {{may only use values defined in its regions or the op's operands}}
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
    // expected-error @+1 {{using value defined outside the region}}
    hipsr.pool_domain_yield %other : tensor<3x4xf32>
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
// Verifier (HasParent): pool_domain_yield outside a pool_domain is rejected.
func.func @yield_without_parent(%in: tensor<3x4xf32>) {
  // expected-error @+1 {{expects parent op 'hipsr.pool_domain'}}
  hipsr.pool_domain_yield %in : tensor<3x4xf32>
}

// -----
// Verifier: the count check also rejects yielding more values than the op has
// results (here 0 results but 1 yielded value).
func.func @more_yields_than_results(%in: tensor<3x4xf32>) {
  // expected-error @+1 {{has 0 result(s) but its pool_domain_yield yields 1 value(s)}}
  hipsr.pool_domain(%in : tensor<3x4xf32>) {
    hipsr.pool_domain_yield %in : tensor<3x4xf32>
  }
  return
}

// -----
// Verifier: a mismatch on a later result is reported with that result's index
// (result #0 matches, result #1 does not).
func.func @later_result_type_mismatch(%in: tensor<3x4xf32>)
    -> (tensor<3x4xf32>, tensor<3x4xi64>) {
  // expected-error @+1 {{result #1 type 'tensor<3x4xi64>' does not match the yielded value type 'tensor<3x4xf32>'}}
  %0:2 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
    hipsr.pool_domain_yield %in, %in : tensor<3x4xf32>, tensor<3x4xf32>
  } -> tensor<3x4xf32>, tensor<3x4xi64>
  return %0#0, %0#1 : tensor<3x4xf32>, tensor<3x4xi64>
}

// -----
// Verifier: types must match exactly -- a tensor result does not match a memref
// yielded value even with the same shape and element type.
func.func @tensor_vs_memref_mismatch(%in: memref<3x4xf32>) -> tensor<3x4xf32> {
  // expected-error @+1 {{result #0 type 'tensor<3x4xf32>' does not match the yielded value type 'memref<3x4xf32>'}}
  %0 = hipsr.pool_domain(%in : memref<3x4xf32>) {
    hipsr.pool_domain_yield %in : memref<3x4xf32>
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
// Verifier: same element type but different static dims is still a mismatch.
func.func @shape_mismatch_static(%in: tensor<4x3xf32>) -> tensor<3x4xf32> {
  // expected-error @+1 {{result #0 type 'tensor<3x4xf32>' does not match the yielded value type 'tensor<4x3xf32>'}}
  %0 = hipsr.pool_domain(%in : tensor<4x3xf32>) {
    hipsr.pool_domain_yield %in : tensor<4x3xf32>
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
// Verifier: differing rank (same total element count) is a mismatch.
func.func @shape_mismatch_rank(%in: tensor<12xf32>) -> tensor<3x4xf32> {
  // expected-error @+1 {{result #0 type 'tensor<3x4xf32>' does not match the yielded value type 'tensor<12xf32>'}}
  %0 = hipsr.pool_domain(%in : tensor<12xf32>) {
    hipsr.pool_domain_yield %in : tensor<12xf32>
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----
// Verifier: a static dim does not match a dynamic dim.
func.func @shape_mismatch_dynamic(%in: tensor<2x?xi64>) -> tensor<2x4xi64> {
  // expected-error @+1 {{result #0 type 'tensor<2x4xi64>' does not match the yielded value type 'tensor<2x?xi64>'}}
  %0 = hipsr.pool_domain(%in : tensor<2x?xi64>) {
    hipsr.pool_domain_yield %in : tensor<2x?xi64>
  } -> tensor<2x4xi64>
  return %0 : tensor<2x4xi64>
}

// -----
// Verifier: memref results follow the same shape-consistency rule.
func.func @shape_mismatch_memref(%in: memref<2x?xi64>) -> memref<2x8xi64> {
  // expected-error @+1 {{result #0 type 'memref<2x8xi64>' does not match the yielded value type 'memref<2x?xi64>'}}
  %0 = hipsr.pool_domain(%in : memref<2x?xi64>) {
    hipsr.pool_domain_yield %in : memref<2x?xi64>
  } -> memref<2x8xi64>
  return %0 : memref<2x8xi64>
}

// -----
// Verifier (SizedRegion<1>): the body must be a single block. Two blocks, each
// terminated by its own pool_domain_yield, is rejected.
func.func @multi_block_body(%in: tensor<3x4xf32>) {
  // expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
  hipsr.pool_domain(%in : tensor<3x4xf32>) {
    hipsr.pool_domain_yield
  ^bb1:
    hipsr.pool_domain_yield
  }
  return
}
