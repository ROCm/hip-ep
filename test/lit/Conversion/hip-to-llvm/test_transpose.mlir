// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  // 3-D transpose with full permutation, static shape, f32.
  func.func @transpose_3d_f32(
      %ctx: !hip.context,
      %input: memref<1x2x3xf32, 1>,
      %output: memref<3x1x2xf32, 1>) {
    hip.transpose(%ctx) ins(%input : memref<1x2x3xf32, 1>)
                        outs(%output : memref<3x1x2xf32, 1>)
                        {perm = [2, 0, 1]}
    return
  }

  // 2-D matrix transpose, static shape, f16.
  func.func @transpose_2d_f16(
      %ctx: !hip.context,
      %input: memref<3x5xf16, 1>,
      %output: memref<5x3xf16, 1>) {
    hip.transpose(%ctx) ins(%input : memref<3x5xf16, 1>)
                        outs(%output : memref<5x3xf16, 1>)
                        {perm = [1, 0]}
    return
  }

  // 4-D transpose (NHWC->NCHW pattern), static shape, f32.
  func.func @transpose_4d_nhwc_to_nchw(
      %ctx: !hip.context,
      %input: memref<1x224x224x3xf32, 1>,
      %output: memref<1x3x224x224xf32, 1>) {
    hip.transpose(%ctx) ins(%input : memref<1x224x224x3xf32, 1>)
                        outs(%output : memref<1x3x224x224xf32, 1>)
                        {perm = [0, 3, 1, 2]}
    return
  }

  // Dynamic outermost dimension, i64 element type.
  func.func @transpose_dynamic_i64(
      %ctx: !hip.context,
      %input: memref<?x4xi64, 1>,
      %output: memref<4x?xi64, 1>) {
    hip.transpose(%ctx) ins(%input : memref<?x4xi64, 1>)
                        outs(%output : memref<4x?xi64, 1>)
                        {perm = [1, 0]}
    return
  }
}

// CHECK-LABEL: llvm.func @transpose_3d_f32
// CHECK: llvm.alloca {{.*}} : (i64) -> !llvm.ptr
// CHECK: llvm.alloca {{.*}} : (i64) -> !llvm.ptr
// CHECK: llvm.call @wrap_transpose({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

// CHECK-LABEL: llvm.func @transpose_2d_f16
// CHECK: llvm.call @wrap_transpose({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

// CHECK-LABEL: llvm.func @transpose_4d_nhwc_to_nchw
// CHECK: llvm.call @wrap_transpose({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

// CHECK-LABEL: llvm.func @transpose_dynamic_i64
// CHECK: llvm.call @wrap_transpose({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
