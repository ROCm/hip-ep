// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @matmul_rank2
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr,
// CHECK:       %[[M:.*]] = llvm.extractvalue {{.*}}[3, 0]
// CHECK-NEXT:  %[[K:.*]] = llvm.extractvalue {{.*}}[3, 1]
// CHECK-NEXT:  %[[N:.*]] = llvm.extractvalue {{.*}}[3, 1]
// CHECK-NEXT:  %[[BATCH:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[B_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  %[[SLOT:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:  %[[A_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[B_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[OUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[ELEM_SIZE:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  llvm.call @wrap_hipblasLtMatmul(%[[CTX]], %[[SLOT]], %[[A_PTR]], %[[B_PTR]], %[[OUT_PTR]], %[[M]], %[[N]], %[[K]], %[[BATCH]], %[[ELEM_SIZE]], %[[B_STRIDE]], {{.*}}, {{.*}}) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
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
// CHECK-NEXT:  %[[DYN_N:.*]] = llvm.extractvalue {{.*}}[3, 2]
// CHECK-NEXT:  %[[BATCH_INIT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[A_BATCH_DIM:.*]] = llvm.extractvalue {{.*}}[3, 0]
// CHECK-NEXT:  %[[DYN_BATCH:.*]] = llvm.mul %[[BATCH_INIT]], %[[A_BATCH_DIM]] : i64
// CHECK-NEXT:  %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  %[[B_BATCH_DIM:.*]] = llvm.extractvalue {{.*}}[3, 0]
// CHECK-NEXT:  %[[LEADING_PRODUCT:.*]] = llvm.mul %[[ONE]], %[[B_BATCH_DIM]] : i64
// CHECK-NEXT:  %[[IS_BROADCAST:.*]] = llvm.icmp "sle" %[[LEADING_PRODUCT]], %[[ONE]] : i64
// CHECK-NEXT:  %[[KN:.*]] = llvm.mul %[[DYN_K]], %[[DYN_N]] : i64
// CHECK-NEXT:  %[[DYN_B_STRIDE:.*]] = llvm.select %[[IS_BROADCAST]], %[[ZERO]], %[[KN]] : i1, i64
// CHECK:       llvm.alloca
// CHECK:       llvm.store
// CHECK:       llvm.alloca
// CHECK:       llvm.store
// CHECK:       llvm.call @wrap_matmul_broadcast({{.*}}) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i32
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

// CHECK-LABEL: llvm.func @matmul_bf16
// CHECK: llvm.mlir.constant(-2 : i64)
// CHECK: llvm.call @wrap_hipblasLtMatmul
func.func @matmul_bf16(%ctx: !hipsr.context, %a: memref<19x37xbf16, #hipsr.mem<device>>, %b: memref<37x21xbf16, #hipsr.mem<device>>, %y: memref<19x21xbf16, #hipsr.mem<device>>) {
  hipsr.matmul(%ctx) ins(%a, %b : memref<19x37xbf16, #hipsr.mem<device>>, memref<37x21xbf16, #hipsr.mem<device>>) outs(%y : memref<19x21xbf16, #hipsr.mem<device>>)
  return
}
// CHECK-LABEL: llvm.func @matmul_partial_broadcast
// CHECK: llvm.alloca
// CHECK: llvm.store
// CHECK: llvm.call @wrap_matmul_broadcast
func.func @matmul_partial_broadcast(%ctx: !hipsr.context, %a: memref<2x3x19x37xf16, #hipsr.mem<device>>, %b: memref<1x3x37x21xf16, #hipsr.mem<device>>, %y: memref<2x3x19x21xf16, #hipsr.mem<device>>) {
  hipsr.matmul(%ctx) ins(%a, %b : memref<2x3x19x37xf16, #hipsr.mem<device>>, memref<1x3x37x21xf16, #hipsr.mem<device>>) outs(%y : memref<2x3x19x21xf16, #hipsr.mem<device>>)
  return
}
// CHECK-LABEL: llvm.func @matmul_cross_broadcast
// CHECK: llvm.alloca
// CHECK: llvm.store
// CHECK: llvm.call @wrap_matmul_broadcast
func.func @matmul_cross_broadcast(%ctx: !hipsr.context, %a: memref<2x1x19x37xf16, #hipsr.mem<device>>, %b: memref<1x3x37x21xf16, #hipsr.mem<device>>, %y: memref<2x3x19x21xf16, #hipsr.mem<device>>) {
  hipsr.matmul(%ctx) ins(%a, %b : memref<2x1x19x37xf16, #hipsr.mem<device>>, memref<1x3x37x21xf16, #hipsr.mem<device>>) outs(%y : memref<2x3x19x21xf16, #hipsr.mem<device>>)
  return
}
// CHECK-LABEL: llvm.func @matmul_unbatched_a
// CHECK: llvm.alloca
// CHECK: llvm.store
// CHECK: llvm.call @wrap_matmul_broadcast
func.func @matmul_unbatched_a(%ctx: !hipsr.context, %a: memref<19x37xf16, #hipsr.mem<device>>, %b: memref<3x37x21xf16, #hipsr.mem<device>>, %y: memref<3x19x21xf16, #hipsr.mem<device>>) {
  hipsr.matmul(%ctx) ins(%a, %b : memref<19x37xf16, #hipsr.mem<device>>, memref<3x37x21xf16, #hipsr.mem<device>>) outs(%y : memref<3x19x21xf16, #hipsr.mem<device>>)
  return
}
