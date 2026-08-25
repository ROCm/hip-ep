// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @floor_static_2d_f32(
      %ctx: !hip.context,
      %x: memref<128x64xf32, 1>,
      %y: memref<128x64xf32, 1>) {
    // CHECK-LABEL: llvm.func @floor_static_2d_f32

    hip.floor(%ctx) ins(%x : memref<128x64xf32, 1>)
                   outs(%y : memref<128x64xf32, 1>)

    // CHECK: llvm.call @wrap_floor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }

  func.func @floor_dynamic_2d_f16(
      %ctx: !hip.context,
      %x: memref<?x?xf16, 1>,
      %y: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @floor_dynamic_2d_f16

    hip.floor(%ctx) ins(%x : memref<?x?xf16, 1>)
                   outs(%y : memref<?x?xf16, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK: llvm.call @wrap_floor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }
}
