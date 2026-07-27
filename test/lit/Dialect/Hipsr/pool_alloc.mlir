// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-pool-alloc | FileCheck %s

// The skeleton pass is a no-op: the pool_domain and its body round-trip
// unchanged.
// CHECK-LABEL: func.func @pool_alloc_noop(
// CHECK-SAME: %[[INPUT:.+]]: tensor<3x4xf32>) -> tensor<2x4xi64> {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.pool_domain(%[[INPUT]] : tensor<3x4xf32>) {
// CHECK-NEXT: ^bb0(%[[DOMAIN_INPUT:.+]]: tensor<3x4xf32>):
// CHECK-NEXT: %[[BUFFER:.+]] = tensor.empty() : tensor<2x4xi64>
// CHECK-NEXT: hipsr.pool_domain_yield %[[BUFFER]] : tensor<2x4xi64>
// CHECK-NEXT: } -> tensor<2x4xi64>
// CHECK-NEXT: return %[[RESULT]] : tensor<2x4xi64>
// CHECK-NEXT: }
func.func @pool_alloc_noop(%in: tensor<3x4xf32>) -> tensor<2x4xi64> {
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
  ^bb0(%domain_in: tensor<3x4xf32>):
    %buf = tensor.empty() : tensor<2x4xi64>
    hipsr.pool_domain_yield %buf : tensor<2x4xi64>
  } -> tensor<2x4xi64>
  return %0 : tensor<2x4xi64>
}
