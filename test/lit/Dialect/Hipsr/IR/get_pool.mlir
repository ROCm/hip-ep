// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// UNSUPPORTED: true

// RUN: hip-mlir-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @get_pool_zero
func.func @get_pool_zero(%ctx: !hipsr.context, %size: index)
    -> memref<?xi8, #hipsr.mem<device>> {
  // CHECK: hipsr.get_pool(%{{.+}}, %{{.+}}) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
  %pool = hipsr.get_pool(%ctx, %size) {domain_id = 0 : i64}
      : memref<?xi8, #hipsr.mem<device>>
  return %pool : memref<?xi8, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: func.func @get_pool_domain
func.func @get_pool_domain(%ctx: !hipsr.context, %size: index)
    -> memref<?xi8, #hipsr.mem<device>> {
  // CHECK: hipsr.get_pool(%{{.+}}, %{{.+}}) {domain_id = 1 : i64} : memref<?xi8, #hipsr.mem<device>>
  %pool = hipsr.get_pool(%ctx, %size) {domain_id = 1 : i64}
      : memref<?xi8, #hipsr.mem<device>>
  return %pool : memref<?xi8, #hipsr.mem<device>>
}
