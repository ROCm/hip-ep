// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// Erase the op.
// CHECK-LABEL: llvm.func @bufferized_shape(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG1:[^,]*]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG2:[^,]*]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG3:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG4:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG5:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG6:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG7:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG8:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG9:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG10:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG11:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG12:[^,]*]]: i64) {
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
func.func @bufferized_shape(%ctx: !hipsr.context, %shape: memref<2xindex>,
                            %data: memref<?x4xf16, #hipsr.mem<device>>) {
  hipsr.preserve_shape %shape, %data
      : memref<2xindex>, memref<?x4xf16, #hipsr.mem<device>>
  return
}
