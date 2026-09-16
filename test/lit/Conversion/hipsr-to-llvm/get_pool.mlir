// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @hipdnn_ep_get_pool_base(!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-LABEL: llvm.func @get_pool(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG1:[^,]*]]: i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> {
// CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V1:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V0]], %[[ARG1]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V3:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V4:.*]] = llvm.insertvalue %[[V1]], %[[V3]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V5:.*]] = llvm.insertvalue %[[V1]], %[[V4]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V6:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V7:.*]] = llvm.insertvalue %[[V6]], %[[V5]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V8:.*]] = llvm.insertvalue %[[ARG1]], %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V9:.*]] = llvm.insertvalue %[[V2]], %[[V8]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.return %[[V9]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:  }
func.func @get_pool(%ctx: !hipsr.context, %size: index)
    -> memref<?xi8, #hipsr.mem<device>> {
  %pool = hipsr.get_pool(%ctx, %size) {domain_id = 0 : i64}
      : memref<?xi8, #hipsr.mem<device>>
  return %pool : memref<?xi8, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: llvm.func @hipdnn_ep_get_pool_base(!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-LABEL: llvm.func @get_pool_domain(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG1:[^,]*]]: i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> {
// CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[V1:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V0]], %[[ARG1]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V3:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V4:.*]] = llvm.insertvalue %[[V1]], %[[V3]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V5:.*]] = llvm.insertvalue %[[V1]], %[[V4]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V6:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V7:.*]] = llvm.insertvalue %[[V6]], %[[V5]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V8:.*]] = llvm.insertvalue %[[ARG1]], %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V9:.*]] = llvm.insertvalue %[[V2]], %[[V8]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.return %[[V9]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:  }
func.func @get_pool_domain(%ctx: !hipsr.context, %size: index)
    -> memref<?xi8, #hipsr.mem<device>> {
  %pool = hipsr.get_pool(%ctx, %size) {domain_id = 2 : i64}
      : memref<?xi8, #hipsr.mem<device>>
  return %pool : memref<?xi8, #hipsr.mem<device>>
}
