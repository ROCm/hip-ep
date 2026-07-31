// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// UNSUPPORTED: true

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics -hipsr-pool-alloc | FileCheck %s

// CHECK-LABEL: func.func @empty_domain
// CHECK-NEXT: hipsr.pool_domain() {
// CHECK-NEXT: }
// CHECK-NOT: hipsr.get_pool
func.func @empty_domain() {
  hipsr.pool_domain() {
    hipsr.pool_domain_yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @noalloc_noop
// CHECK: hipsr.pool_domain
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<2x4xi64>
// CHECK-NEXT: hipsr.pool_domain_yield %[[BUF]] : tensor<2x4xi64>
// CHECK-NOT: hipsr.get_pool
// CHECK-NOT: memref.view
func.func @noalloc_noop(%ctx: !hipsr.context,
                        %in: tensor<3x4xf32>) -> tensor<2x4xi64> {
  %0 = hipsr.pool_domain(%ctx, %in : !hipsr.context, tensor<3x4xf32>) {
  ^bb0(%dctx: !hipsr.context, %din: tensor<3x4xf32>):
    %buf = tensor.empty() : tensor<2x4xi64>
    hipsr.pool_domain_yield %buf : tensor<2x4xi64>
  } -> tensor<2x4xi64>
  return %0 : tensor<2x4xi64>
}
