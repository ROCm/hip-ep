// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @hipdnn_ep_alloc_output(!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-LABEL: llvm.func @alloc_output_static(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> {
// CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V1:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V2:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V3:.*]] = llvm.mlir.constant(6 : index) : i64
// CHECK-NEXT:    %[[V4:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V5:.*]] = llvm.getelementptr %[[V4]][%[[V3]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:    %[[V6:.*]] = llvm.ptrtoint %[[V5]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V7:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V8:.*]] = llvm.alloca %[[V7]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V9:.*]] = llvm.getelementptr %[[V8]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V0]], %[[V9]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V10:.*]] = llvm.getelementptr %[[V8]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1]], %[[V10]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V11:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V12:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V13:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V14:.*]] = llvm.call @hipdnn_ep_alloc_output(%[[ARG0]], %[[V11]], %[[V8]], %[[V12]], %[[V13]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V15:.*]] = llvm.addrspacecast %[[V14]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[V16:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V17:.*]] = llvm.insertvalue %[[V15]], %[[V16]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V18:.*]] = llvm.insertvalue %[[V15]], %[[V17]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V19:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V20:.*]] = llvm.insertvalue %[[V19]], %[[V18]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V21:.*]] = llvm.insertvalue %[[V0]], %[[V20]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V22:.*]] = llvm.insertvalue %[[V1]], %[[V21]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V23:.*]] = llvm.insertvalue %[[V1]], %[[V22]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V24:.*]] = llvm.insertvalue %[[V2]], %[[V23]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    llvm.return %[[V24]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  }
func.func @alloc_output_static(%ctx: !hipsr.context)
    -> memref<2x3xf32, #hipsr.mem<device>> {
  %out = hipsr.alloc_output(%ctx) {out_idx = 0 : i64}
      : memref<2x3xf32, #hipsr.mem<device>>
  return %out : memref<2x3xf32, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: llvm.func @hipdnn_ep_alloc_output(!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-LABEL: llvm.func @alloc_output_dynamic(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG1:[^,]*]]: i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> {
// CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK-NEXT:    %[[V1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V2:.*]] = llvm.mul %[[V0]], %[[ARG1]] : i64
// CHECK-NEXT:    %[[V3:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V4:.*]] = llvm.getelementptr %[[V3]][%[[V2]]] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-NEXT:    %[[V5:.*]] = llvm.ptrtoint %[[V4]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V6:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V7:.*]] = llvm.alloca %[[V6]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V8:.*]] = llvm.getelementptr %[[V7]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[ARG1]], %[[V8]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V9:.*]] = llvm.getelementptr %[[V7]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V0]], %[[V9]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V10:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V11:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V12:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V13:.*]] = llvm.call @hipdnn_ep_alloc_output(%[[ARG0]], %[[V10]], %[[V7]], %[[V11]], %[[V12]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V14:.*]] = llvm.addrspacecast %[[V13]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[V15:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V16:.*]] = llvm.insertvalue %[[V14]], %[[V15]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V17:.*]] = llvm.insertvalue %[[V14]], %[[V16]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V18:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V19:.*]] = llvm.insertvalue %[[V18]], %[[V17]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V20:.*]] = llvm.insertvalue %[[ARG1]], %[[V19]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V21:.*]] = llvm.insertvalue %[[V0]], %[[V20]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V22:.*]] = llvm.insertvalue %[[V0]], %[[V21]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V23:.*]] = llvm.insertvalue %[[V1]], %[[V22]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    llvm.return %[[V23]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  }
func.func @alloc_output_dynamic(%ctx: !hipsr.context, %n: index)
    -> memref<?x8xf16, #hipsr.mem<device>> {
  %out = hipsr.alloc_output(%ctx, %n) {out_idx = 1 : i64}
      : memref<?x8xf16, #hipsr.mem<device>>
  return %out : memref<?x8xf16, #hipsr.mem<device>>
}
