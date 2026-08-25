// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// The call's data type is 4 (i64), the operand type being compared. The mask
// needs none of its own: the runtime always writes one byte per element.
// CHECK-LABEL: llvm.func @equal
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
// CHECK-NEXT:  %[[LHS_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[RHS_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[OUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[DATA_TYPE:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:  llvm.call @wrap_equal(%[[CTX]], %[[LHS_PTR]], %[[RHS_PTR]], %[[OUT_PTR]], %[[LHS_N]], %[[LHS_C]], %[[LHS_H]], %[[LHS_W]], %[[RHS_N]], %[[RHS_C]], %[[RHS_H]], %[[RHS_W]], %[[OUT_N]], %[[OUT_C]], %[[OUT_H]], %[[OUT_W]], %[[DATA_TYPE]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.return
func.func @equal(%ctx: !hipsr.context,
                 %lhs: memref<4x1024xi64, #hipsr.mem<device>>,
                 %rhs: memref<1024xi64, #hipsr.mem<device>>,
                 %init: memref<4x1024xui8, #hipsr.mem<device>>) {
  hipsr.equal(%ctx) ins(%lhs, %rhs : memref<4x1024xi64, #hipsr.mem<device>>, memref<1024xi64, #hipsr.mem<device>>)
               outs(%init : memref<4x1024xui8, #hipsr.mem<device>>)
  return
}
