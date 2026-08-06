// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// UNSUPPORTED: true

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK: llvm.func @hipdnn_ep_get_pool_base(!llvm.ptr, i32, i64) -> !llvm.ptr<1>

// CHECK-LABEL: llvm.func @get_pool
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr, %[[SIZE:.*]]: i64)
// CHECK-NEXT:    %[[DOMAIN:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[PTR:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[CTX]], %[[DOMAIN]], %[[SIZE]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[STRIDE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[D0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[D1:.*]] = llvm.insertvalue %[[PTR]], %[[D0]][0]
// CHECK-NEXT:    %[[D2:.*]] = llvm.insertvalue %[[PTR]], %[[D1]][1]
// CHECK-NEXT:    %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[D3:.*]] = llvm.insertvalue %[[OFFSET]], %[[D2]][2]
// CHECK-NEXT:    %[[D4:.*]] = llvm.insertvalue %[[SIZE]], %[[D3]][3, 0]
// CHECK-NEXT:    %[[D5:.*]] = llvm.insertvalue %[[STRIDE]], %[[D4]][4, 0]
// CHECK-NEXT:    llvm.return %[[D5]]
func.func @get_pool(%ctx: !hipsr.context, %size: index)
    -> memref<?xi8, #hipsr.mem<device>> {
  %pool = hipsr.get_pool(%ctx, %size) {domain_id = 0 : i64}
      : memref<?xi8, #hipsr.mem<device>>
  return %pool : memref<?xi8, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: llvm.func @get_pool_domain
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr, %[[SIZE:.*]]: i64)
// CHECK-NEXT:    %[[DOMAIN:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[PTR:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[CTX]], %[[DOMAIN]], %[[SIZE]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
func.func @get_pool_domain(%ctx: !hipsr.context, %size: index)
    -> memref<?xi8, #hipsr.mem<device>> {
  %pool = hipsr.get_pool(%ctx, %size) {domain_id = 2 : i64}
      : memref<?xi8, #hipsr.mem<device>>
  return %pool : memref<?xi8, #hipsr.mem<device>>
}
