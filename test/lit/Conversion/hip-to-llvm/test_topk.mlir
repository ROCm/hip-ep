// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @topk_static_2d_f32(
      %ctx: !hip.context,
      %x: memref<3x4xf32, 1>,
      %k: memref<i64, 1>,
      %values: memref<3x2xf32, 1>,
      %indices: memref<3x2xi64, 1>) {
    // CHECK-LABEL: llvm.func @topk_static_2d_f32

    hip.top_k(%ctx)
        ins(%x, %k : memref<3x4xf32, 1>, memref<i64, 1>)
        outs(%values, %indices : memref<3x2xf32, 1>, memref<3x2xi64, 1>)
        {axis = 1 : i64, largest = true, sorted = true}

    // CHECK: llvm.call @wrap_top_k({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, i64) -> i32
    return
  }

  func.func @topk_dynamic_2d_f16(
      %ctx: !hip.context,
      %x: memref<?x?xf16, 1>,
      %k: memref<1xi64, 1>,
      %values: memref<?x?xf16, 1>,
      %indices: memref<?x?xi64, 1>) {
    // CHECK-LABEL: llvm.func @topk_dynamic_2d_f16

    hip.top_k(%ctx)
        ins(%x, %k : memref<?x?xf16, 1>, memref<1xi64, 1>)
        outs(%values, %indices : memref<?x?xf16, 1>, memref<?x?xi64, 1>)
        {axis = 1 : i64, largest = true, sorted = true}

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_top_k({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, i64) -> i32
    return
  }
}
