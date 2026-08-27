// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @expand_host_shape
// CHECK-SAME:  (%[[CTX:[^,]+]]: !llvm.ptr,
// CHECK:       %[[SHAPE_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr,
// CHECK:       %[[INPUT_SHAPE:.*]] = llvm.alloca {{.*}} x !llvm.array<2 x i64>
// CHECK:       %[[OUTPUT_SHAPE:.*]] = llvm.alloca {{.*}} x !llvm.array<3 x i64>
// CHECK:       %[[INPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[OUTPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[INPUT_RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  %[[OUTPUT_RANK:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:  %[[DATA_TYPE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  llvm.call @wrap_expand(%[[CTX]], %[[INPUT_PTR]], %[[SHAPE_PTR]], %[[OUTPUT_PTR]], %[[INPUT_SHAPE]], %[[INPUT_RANK]], %[[OUTPUT_SHAPE]], %[[OUTPUT_RANK]], %[[DATA_TYPE]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
func.func @expand_host_shape(
    %ctx: !hipsr.context,
    %input: memref<3x1xf32, #hipsr.mem<device>>,
    %shape: memref<3xi64, #hipsr.mem<host>>,
    %init: memref<2x3x6xf32, #hipsr.mem<device>>) {
  hipsr.expand(%ctx)
      ins(%input, %shape : memref<3x1xf32, #hipsr.mem<device>>,
                            memref<3xi64, #hipsr.mem<host>>)
      outs(%init : memref<2x3x6xf32, #hipsr.mem<device>>)
  return
}

// CHECK-LABEL: llvm.func @expand_shape_attr
// CHECK-SAME:  (%[[ATTR_CTX:[^,]+]]: !llvm.ptr,
// CHECK:       %[[NULL_SHAPE:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:       %[[ATTR_INPUT_SHAPE:.*]] = llvm.alloca {{.*}} x !llvm.array<2 x i64>
// CHECK:       %[[ATTR_OUTPUT_SHAPE:.*]] = llvm.alloca {{.*}} x !llvm.array<3 x i64>
// CHECK:       llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>,
// CHECK:       %[[ATTR_INPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[ATTR_OUTPUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[ATTR_INPUT_RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  %[[ATTR_OUTPUT_RANK:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:  %[[ATTR_DATA_TYPE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  llvm.call @wrap_expand(%[[ATTR_CTX]], %[[ATTR_INPUT_PTR]], %[[NULL_SHAPE]], %[[ATTR_OUTPUT_PTR]], %[[ATTR_INPUT_SHAPE]], %[[ATTR_INPUT_RANK]], %[[ATTR_OUTPUT_SHAPE]], %[[ATTR_OUTPUT_RANK]], %[[ATTR_DATA_TYPE]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
func.func @expand_shape_attr(
    %ctx: !hipsr.context,
    %input: memref<?x1xf16, #hipsr.mem<device>>,
    %init: memref<?x3x?xf16, #hipsr.mem<device>>) {
  hipsr.expand(%ctx)
      ins(%input : memref<?x1xf16, #hipsr.mem<device>>)
      outs(%init : memref<?x3x?xf16, #hipsr.mem<device>>)
      {shape_attr = array<i64: 2, 3, 6>}
  return
}
