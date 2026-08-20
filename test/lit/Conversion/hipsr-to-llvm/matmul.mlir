// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @matmul_rank2
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr,
// CHECK:       %[[M:.*]] = llvm.extractvalue {{.*}}[3, 0]
// CHECK-NEXT:  %[[K:.*]] = llvm.extractvalue {{.*}}[3, 1]
// CHECK-NEXT:  %[[B_K:.*]] = llvm.extractvalue {{.*}}[3, 0]
// CHECK-NEXT:  %[[N:.*]] = llvm.extractvalue {{.*}}[3, 1]
// CHECK-NEXT:  %[[BATCH:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[B_COUNT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[B_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  %[[VALID:.*]] = llvm.mlir.constant({{.*}}) : i1
// CHECK-NEXT:  %[[A_STRIDE:.*]] = llvm.mul %[[M]], %[[K]] : i64
// CHECK-NEXT:  %[[SLOT:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:  %[[A_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[B_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[OUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[ELEM_SIZE:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  llvm.call @wrap_hipblasLtMatmul(%[[CTX]], %[[SLOT]], %[[A_PTR]], %[[B_PTR]], %[[OUT_PTR]], %[[VALID]], %[[M]], %[[N]], %[[K]], %[[B_K]], %[[BATCH]], %[[ELEM_SIZE]], %[[BATCH]], %[[B_COUNT]], %[[A_STRIDE]], %[[B_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i1, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.return
func.func @matmul_rank2(
    %ctx: !hipsr.context,
    %a: memref<128x64xf16, #hipsr.mem<device>>,
    %b: memref<64x32xf16, #hipsr.mem<device>>,
    %init: memref<128x32xf16, #hipsr.mem<device>>) {
  hipsr.matmul(%ctx)
      ins(%a, %b : memref<128x64xf16, #hipsr.mem<device>>,
                    memref<64x32xf16, #hipsr.mem<device>>)
      outs(%init : memref<128x32xf16, #hipsr.mem<device>>)
  return
}

// CHECK-LABEL: llvm.func @matmul_dynamic_batch
// CHECK-SAME:  (%[[DYN_CTX:.*]]: !llvm.ptr,
// CHECK:       %[[DYN_M:.*]] = llvm.extractvalue {{.*}}[3, 1]
// CHECK-NEXT:  %[[DYN_K:.*]] = llvm.extractvalue {{.*}}[3, 2]
// CHECK-NEXT:  %[[DYN_B_K:.*]] = llvm.extractvalue {{.*}}[3, 1]
// CHECK-NEXT:  %[[DYN_N:.*]] = llvm.extractvalue {{.*}}[3, 2]
// CHECK-NEXT:  %[[BATCH_INIT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[A_BATCH_DIM:.*]] = llvm.extractvalue {{.*}}[3, 0]
// CHECK-NEXT:  %[[DYN_BATCH:.*]] = llvm.mul %[[BATCH_INIT]], %[[A_BATCH_DIM]] : i64
// CHECK-NEXT:  %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  %[[B_BATCH_DIM:.*]] = llvm.extractvalue {{.*}}[3, 0]
// CHECK-NEXT:  %[[B_COUNT:.*]] = llvm.mul %[[ONE]], %[[B_BATCH_DIM]] : i64
// CHECK-NEXT:  %[[IS_BROADCAST:.*]] = llvm.icmp "sle" %[[B_COUNT]], %[[ONE]] : i64
// CHECK-NEXT:  %[[KN:.*]] = llvm.mul %[[DYN_B_K]], %[[DYN_N]] : i64
// CHECK-NEXT:  %[[DYN_B_STRIDE:.*]] = llvm.select %[[IS_BROADCAST]], %[[ZERO]], %[[KN]] : i1, i64
// CHECK-NEXT:  %[[DYN_VALID:.*]] = llvm.mlir.constant({{.*}}) : i1
// CHECK-NEXT:  %[[DYN_A_STRIDE:.*]] = llvm.mul %[[DYN_M]], %[[DYN_K]] : i64
// CHECK-NEXT:  %[[DYN_SLOT:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:  %[[DYN_A_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[DYN_B_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[DYN_OUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[DYN_ELEM_SIZE:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  llvm.call @wrap_hipblasLtMatmul(%[[DYN_CTX]], %[[DYN_SLOT]], %[[DYN_A_PTR]], %[[DYN_B_PTR]], %[[DYN_OUT_PTR]], %[[DYN_VALID]], %[[DYN_M]], %[[DYN_N]], %[[DYN_K]], %[[DYN_B_K]], %[[DYN_BATCH]], %[[DYN_ELEM_SIZE]], %[[DYN_BATCH]], %[[B_COUNT]], %[[DYN_A_STRIDE]], %[[DYN_B_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i1, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.return
func.func @matmul_dynamic_batch(
    %ctx: !hipsr.context,
    %a: memref<?x128x64xf16, #hipsr.mem<device>>,
    %b: memref<?x64x32xf16, #hipsr.mem<device>>,
    %init: memref<?x128x32xf16, #hipsr.mem<device>>) {
  hipsr.matmul(%ctx)
      ins(%a, %b : memref<?x128x64xf16, #hipsr.mem<device>>,
                    memref<?x64x32xf16, #hipsr.mem<device>>)
      outs(%init : memref<?x128x32xf16, #hipsr.mem<device>>)
  return
}
