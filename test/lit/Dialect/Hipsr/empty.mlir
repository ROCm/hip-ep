// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Tests that hipsr.empty / hipsr.empty_yield parse, round-trip, and that the
// verifiers reject malformed IR.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// -----
// Round-trip: single result with a dynamic dim. The (non-isolated) region reads
// a parent-scope value (%input) to compute the shape.
// CHECK-LABEL: func.func @roundtrip_single
// CHECK: %[[R:.*]] = hipsr.empty() : tensor<?x8xf16> {
// CHECK:   %[[SH:.*]] = shape.shape_of
// CHECK:   %[[D0:.*]] = shape.get_extent
// CHECK:   %[[T:.*]] = tensor.empty(%[[D0]]) : tensor<?x8xf16>
// CHECK:   hipsr.empty_yield %[[T]] : tensor<?x8xf16>
// CHECK: }
func.func @roundtrip_single(%input: tensor<?x8xf32>) -> tensor<?x8xf16> {
  %0 = hipsr.empty() : tensor<?x8xf16> {
    %shape = shape.shape_of %input : tensor<?x8xf32> -> tensor<2xindex>
    %c0 = arith.constant 0 : index
    %d0 = shape.get_extent %shape, %c0 : tensor<2xindex>, index -> index
    %t = tensor.empty(%d0) : tensor<?x8xf16>
    hipsr.empty_yield %t : tensor<?x8xf16>
  }
  return %0 : tensor<?x8xf16>
}

// -----
// Round-trip: multiple results (a multi-output DPS op's inits).
// CHECK-LABEL: func.func @roundtrip_multi
// CHECK: %[[R:.*]]:2 = hipsr.empty() : tensor<?x?xf16>, tensor<?xi64> {
// CHECK:   hipsr.empty_yield %{{.*}}, %{{.*}} : tensor<?x?xf16>, tensor<?xi64>
// CHECK: }
func.func @roundtrip_multi(%n: index, %m: index) -> (tensor<?x?xf16>, tensor<?xi64>) {
  %0:2 = hipsr.empty() : tensor<?x?xf16>, tensor<?xi64> {
    %t0 = tensor.empty(%n, %m) : tensor<?x?xf16>
    %t1 = tensor.empty(%n) : tensor<?xi64>
    hipsr.empty_yield %t0, %t1 : tensor<?x?xf16>, tensor<?xi64>
  }
  return %0#0, %0#1 : tensor<?x?xf16>, tensor<?xi64>
}

// -----
// Verifier: result count must match the yielded tensor count.
func.func @result_count_mismatch() -> tensor<4xf16> {
  // expected-error @+1 {{has 1 result(s) but its empty_yield yields 2 value(s)}}
  %0 = hipsr.empty() : tensor<4xf16> {
    %t0 = tensor.empty() : tensor<4xf16>
    %t1 = tensor.empty() : tensor<4xf16>
    hipsr.empty_yield %t0, %t1 : tensor<4xf16>, tensor<4xf16>
  }
  return %0 : tensor<4xf16>
}

// -----
// Verifier: result type must match the yielded tensor type.
func.func @result_type_mismatch() -> tensor<4xi64> {
  // expected-error @+1 {{result #0 type 'tensor<4xi64>' does not match the yielded value type 'tensor<4xf16>'}}
  %0 = hipsr.empty() : tensor<4xi64> {
    %t = tensor.empty() : tensor<4xf16>
    hipsr.empty_yield %t : tensor<4xf16>
  }
  return %0 : tensor<4xi64>
}

// -----
// Verifier: a mismatch on a later result is reported with that result's index
// (result #0 matches, result #1 does not).
func.func @later_result_type_mismatch() -> (tensor<4xf16>, tensor<4xi64>) {
  // expected-error @+1 {{result #1 type 'tensor<4xi64>' does not match the yielded value type 'tensor<4xf16>'}}
  %0:2 = hipsr.empty() : tensor<4xf16>, tensor<4xi64> {
    %t0 = tensor.empty() : tensor<4xf16>
    %t1 = tensor.empty() : tensor<4xf16>
    hipsr.empty_yield %t0, %t1 : tensor<4xf16>, tensor<4xf16>
  }
  return %0#0, %0#1 : tensor<4xf16>, tensor<4xi64>
}

// -----
// Verifier: each empty_yield operand must be a tensor.empty result. Here the
// region yields a parent-scope value (%arg) instead of a tensor.empty.
func.func @yield_non_tensor_empty(%arg: tensor<4xf16>) -> tensor<4xf16> {
  %0 = hipsr.empty() : tensor<4xf16> {
    // expected-error @+1 {{operand #0 must be a tensor.empty result}}
    hipsr.empty_yield %arg : tensor<4xf16>
  }
  return %0 : tensor<4xf16>
}

// -----
// Verifier (HasParent): empty_yield outside a hipsr.empty is rejected.
func.func @yield_without_parent(%t: tensor<4xf16>) {
  // expected-error @+1 {{expects parent op 'hipsr.empty'}}
  hipsr.empty_yield %t : tensor<4xf16>
  return
}

// -----
// Verifier (SizedRegion<1>): the region must be a single block. Two blocks,
// each terminated by its own empty_yield, is rejected.
func.func @multi_block_body() -> tensor<4xf16> {
  // expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
  %0 = hipsr.empty() : tensor<4xf16> {
    %t = tensor.empty() : tensor<4xf16>
    hipsr.empty_yield %t : tensor<4xf16>
  ^bb1:
    %t2 = tensor.empty() : tensor<4xf16>
    hipsr.empty_yield %t2 : tensor<4xf16>
  }
  return %0 : tensor<4xf16>
}
