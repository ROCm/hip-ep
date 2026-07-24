// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @onehot_axis1_f32(
      %ctx: !hip.context,
      %indices: memref<2x2xi64, 1>,
      %depth: memref<i64, 1>,
      %values: memref<2xf32, 1>,
      %output: memref<2x10x2xf32, 1>) {
    // CHECK-LABEL: llvm.func @onehot_axis1_f32

    hip.one_hot(%ctx)
        ins(%indices, %depth, %values :
            memref<2x2xi64, 1>, memref<i64, 1>, memref<2xf32, 1>)
        outs(%output : memref<2x10x2xf32, 1>) {axis = 1 : i64}

    // CHECK: llvm.call @wrap_one_hot({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    return
  }
}
