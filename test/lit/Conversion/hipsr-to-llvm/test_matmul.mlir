// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @matmul_rank2
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr,
// CHECK:       %[[A_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK:       %[[B_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK:       %[[OUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK:       llvm.call @wrap_hipblasLtMatmul(%[[CTX]], {{%.*}}, %[[A_PTR]], %[[B_PTR]], %[[OUT_PTR]],
// CHECK-SAME:    (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64) -> i32
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
// CHECK:       %[[DYN_BATCH:.*]] = llvm.mul
// CHECK:       %[[DYN_B_STRIDE:.*]] = llvm.select
// CHECK:       llvm.call @wrap_hipblasLtMatmul(%[[DYN_CTX]], {{%.*}}, {{.*}}, %[[DYN_BATCH]], {{%.*}}, %[[DYN_B_STRIDE]])
// CHECK-SAME:    (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64) -> i32
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
