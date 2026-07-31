// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// UNSUPPORTED: true

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @cast_static
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr,
// CHECK:       %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[DIM_0:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:  %[[COUNT_0:.*]] = llvm.mul %[[ONE]], %[[DIM_0]] : i64
// CHECK-NEXT:  %[[DIM_1:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:  %[[NUM_ELEMENTS:.*]] = llvm.mul %[[COUNT_0]], %[[DIM_1]] : i64
// CHECK-NEXT:  %[[INPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[OUTPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[SRC_TYPE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  %[[DST_TYPE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  llvm.call @wrap_cast(%[[CTX]], %[[INPUT_PTR]], %[[OUTPUT_PTR]], %[[NUM_ELEMENTS]], %[[SRC_TYPE]], %[[DST_TYPE]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.return
func.func @cast_static(
    %ctx: !hipsr.context,
    %input: memref<4x8xf32, #hipsr.mem<device>>,
    %init: memref<4x8xf16, #hipsr.mem<device>>) {
  hipsr.cast(%ctx)
      ins(%input : memref<4x8xf32, #hipsr.mem<device>>)
      outs(%init : memref<4x8xf16, #hipsr.mem<device>>)
  return
}

// CHECK-LABEL: llvm.func @cast_dynamic
// CHECK-SAME:  (%[[DYN_CTX:.*]]: !llvm.ptr,
// CHECK:       %[[DYN_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[DYN_DIM_0:.*]] = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[DYN_COUNT_0:.*]] = llvm.mul %[[DYN_ONE]], %[[DYN_DIM_0]] : i64
// CHECK-NEXT:  %[[DYN_DIM_1:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:  %[[DYN_NUM_ELEMENTS:.*]] = llvm.mul %[[DYN_COUNT_0]], %[[DYN_DIM_1]] : i64
// CHECK-NEXT:  %[[DYN_INPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[DYN_OUTPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[DYN_SRC_TYPE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[DYN_DST_TYPE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  llvm.call @wrap_cast(%[[DYN_CTX]], %[[DYN_INPUT_PTR]], %[[DYN_OUTPUT_PTR]], %[[DYN_NUM_ELEMENTS]], %[[DYN_SRC_TYPE]], %[[DYN_DST_TYPE]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.return
func.func @cast_dynamic(
    %ctx: !hipsr.context,
    %input: memref<?x8xf16, #hipsr.mem<device>>,
    %init: memref<?x8xf32, #hipsr.mem<device>>) {
  hipsr.cast(%ctx)
      ins(%input : memref<?x8xf16, #hipsr.mem<device>>)
      outs(%init : memref<?x8xf32, #hipsr.mem<device>>)
  return
}
