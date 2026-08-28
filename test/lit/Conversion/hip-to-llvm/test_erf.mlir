// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @erf_static_f32(
      %ctx: !hip.context,
      %x: memref<256xf32, 1>,
      %y: memref<256xf32, 1>) {
    // CHECK-LABEL: llvm.func @erf_static_f32
    hip.erf(%ctx) ins(%x : memref<256xf32, 1>)
                  outs(%y : memref<256xf32, 1>)
    // CHECK: llvm.call @wrap_erf({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }

  func.func @erf_dynamic_f16(
      %ctx: !hip.context,
      %x: memref<?x?xf16, 1>,
      %y: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @erf_dynamic_f16
    hip.erf(%ctx) ins(%x : memref<?x?xf16, 1>)
                  outs(%y : memref<?x?xf16, 1>)
    // CHECK: llvm.call @wrap_erf({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }
}
