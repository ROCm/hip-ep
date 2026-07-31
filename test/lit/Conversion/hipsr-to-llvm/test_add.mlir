// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// REQUIRES: hipsr

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @add
// CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr,
// CHECK-DAG:   %[[SLOT:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-DAG:   %[[TOP:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:       llvm.call @wrap_miopenOpTensor(%[[CTX]], %[[SLOT]],
// CHECK-SAME:    !llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
func.func @add(%ctx: !hipsr.context,
               %lhs: memref<4x1024xf16, #hipsr.mem<device>>,
               %rhs: memref<1024xf16, #hipsr.mem<device>>,
               %init: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.add(%ctx) ins(%lhs, %rhs : memref<4x1024xf16, #hipsr.mem<device>>, memref<1024xf16, #hipsr.mem<device>>)
             outs(%init : memref<4x1024xf16, #hipsr.mem<device>>)
  return
}
