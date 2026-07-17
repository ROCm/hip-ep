// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Tests for hipsr.placeholder:
//   - round-trips single-result and multi-result forms
//   - an unranked result is rejected by the AnyRankedTensor constraint
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// -----
// Single result, dynamic dims.
// CHECK-LABEL: func.func @single_dynamic
// CHECK: %{{.*}} = hipsr.placeholder : tensor<?x?xf16>
func.func @single_dynamic() -> tensor<?x?xf16> {
  %0 = hipsr.placeholder : tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----
// Single result, static dims.
// CHECK-LABEL: func.func @single_static
// CHECK: %{{.*}} = hipsr.placeholder : tensor<4x8xi64>
func.func @single_static() -> tensor<4x8xi64> {
  %0 = hipsr.placeholder : tensor<4x8xi64>
  return %0 : tensor<4x8xi64>
}

// -----
// Rank-0 (scalar) result.
// CHECK-LABEL: func.func @rank0
// CHECK: %{{.*}} = hipsr.placeholder : tensor<f32>
func.func @rank0() -> tensor<f32> {
  %0 = hipsr.placeholder : tensor<f32>
  return %0 : tensor<f32>
}

// -----
// Multi-result (variadic), mirroring a multi-output DPS op.
// CHECK-LABEL: func.func @multi_result
// CHECK: %{{.*}}:2 = hipsr.placeholder : tensor<?x?xf16>, tensor<?xi64>
func.func @multi_result() -> (tensor<?x?xf16>, tensor<?xi64>) {
  %0:2 = hipsr.placeholder : tensor<?x?xf16>, tensor<?xi64>
  return %0#0, %0#1 : tensor<?x?xf16>, tensor<?xi64>
}

// -----
// An unranked result is rejected by the AnyRankedTensor constraint.
func.func @unranked_rejected() {
  // expected-error @+1 {{op result #0 must be variadic of ranked tensor of any type values}}
  %0 = "hipsr.placeholder"() : () -> tensor<*xf16>
  return
}
