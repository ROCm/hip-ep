// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @get_pool
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr,
// CHECK:       %[[DOMAIN:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       llvm.call @hipdnn_ep_get_pool_base(%[[CTX]], %[[DOMAIN]], %{{.*}}) : (!llvm.ptr, i32, i64) -> !llvm.ptr
func.func @get_pool(%ctx: !hipsr.context, %size: index)
    -> memref<?xi8, #hipsr.mem<device>> {
  %pool = hipsr.get_pool(%ctx, %size) : memref<?xi8, #hipsr.mem<device>>
  return %pool : memref<?xi8, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: llvm.func @get_pool_domain
// CHECK:       %[[DOMAIN:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:       llvm.call @hipdnn_ep_get_pool_base(%{{.*}}, %[[DOMAIN]], %{{.*}}) : (!llvm.ptr, i32, i64) -> !llvm.ptr
func.func @get_pool_domain(%ctx: !hipsr.context, %size: index)
    -> memref<?xi8, #hipsr.mem<device>> {
  %pool = hipsr.get_pool(%ctx, %size) {domain_id = 2 : i64}
      : memref<?xi8, #hipsr.mem<device>>
  return %pool : memref<?xi8, #hipsr.mem<device>>
}
