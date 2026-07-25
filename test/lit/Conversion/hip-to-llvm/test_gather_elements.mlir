// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @gather_elements_static_2d_f32(
      %ctx: !hip.context,
      %data: memref<2x2xf32, 1>,
      %indices: memref<2x2xi32, 1>,
      %output: memref<2x2xf32, 1>) {
    // CHECK-LABEL: llvm.func @gather_elements_static_2d_f32

    hip.gather_elements(%ctx)
        ins(%data, %indices : memref<2x2xf32, 1>, memref<2x2xi32, 1>)
        outs(%output : memref<2x2xf32, 1>) {axis = 1 : i64}

    // CHECK: llvm.call @wrap_gather_elements({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    return
  }

  func.func @gather_elements_dynamic_2d_f16(
      %ctx: !hip.context,
      %data: memref<?x?xf16, 1>,
      %indices: memref<?x?xi32, 1>,
      %output: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @gather_elements_dynamic_2d_f16

    hip.gather_elements(%ctx)
        ins(%data, %indices : memref<?x?xf16, 1>, memref<?x?xi32, 1>)
        outs(%output : memref<?x?xf16, 1>) {axis = -1 : i64}

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.call @wrap_gather_elements({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    return
  }
}
