// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @mul
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr,
// CHECK:       %[[LHS_N:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[LHS_C:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[LHS_H:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:  %[[LHS_W:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK-NEXT:  %[[RHS_N:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[RHS_C:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[RHS_H:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[RHS_W:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK-NEXT:  %[[OUT_N:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[OUT_C:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[OUT_H:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:  %[[OUT_W:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK-NEXT:  %[[SLOT:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:  %[[LHS_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[RHS_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[OUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[DATA_TYPE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[TENSOR_OP:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  llvm.call @wrap_elementwise(%[[CTX]], %[[SLOT]], %[[LHS_PTR]], %[[RHS_PTR]], %[[OUT_PTR]], %[[LHS_N]], %[[LHS_C]], %[[LHS_H]], %[[LHS_W]], %[[RHS_N]], %[[RHS_C]], %[[RHS_H]], %[[RHS_W]], %[[OUT_N]], %[[OUT_C]], %[[OUT_H]], %[[OUT_W]], %[[DATA_TYPE]], %[[TENSOR_OP]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.return
func.func @mul(%ctx: !hipsr.context,
               %lhs: memref<4x1024xf16, #hipsr.mem<device>>,
               %rhs: memref<1024xf16, #hipsr.mem<device>>,
               %init: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.mul(%ctx) ins(%lhs, %rhs : memref<4x1024xf16, #hipsr.mem<device>>, memref<1024xf16, #hipsr.mem<device>>)
             outs(%init : memref<4x1024xf16, #hipsr.mem<device>>)
  return
}
