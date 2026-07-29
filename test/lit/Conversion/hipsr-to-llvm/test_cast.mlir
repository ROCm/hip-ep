// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @cast_static
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr,
// CHECK:       %[[INPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK:       %[[OUTPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK:       %[[SRC_TYPE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[DST_TYPE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:       llvm.call @wrap_cast(%[[CTX]], %[[INPUT_PTR]], %[[OUTPUT_PTR]], {{%.*}}, %[[SRC_TYPE]], %[[DST_TYPE]])
// CHECK-SAME:    (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
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
// CHECK:       llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<1>,
// CHECK:       %[[DYN_INPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK:       %[[DYN_OUTPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK:       %[[DYN_SRC_TYPE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:       %[[DYN_DST_TYPE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       llvm.call @wrap_cast(%[[DYN_CTX]], %[[DYN_INPUT_PTR]], %[[DYN_OUTPUT_PTR]], {{%.*}}, %[[DYN_SRC_TYPE]], %[[DYN_DST_TYPE]])
// CHECK-SAME:    (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
func.func @cast_dynamic(
    %ctx: !hipsr.context,
    %input: memref<?x8xf16, #hipsr.mem<device>>,
    %init: memref<?x8xf32, #hipsr.mem<device>>) {
  hipsr.cast(%ctx)
      ins(%input : memref<?x8xf16, #hipsr.mem<device>>)
      outs(%init : memref<?x8xf32, #hipsr.mem<device>>)
  return
}
