// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  // Dynamic rank-3 output. The three sizes are loaded from the memref
  // descriptor and stored into the shape array passed to the runtime.
  func.func @rnl_dynamic_3d(
      %ctx: !hip.context,
      %input: memref<?x?x?xf16, 1>,
      %output: memref<?x?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @rnl_dynamic_3d
    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.extractvalue %{{.*}}[3, 2]
    // CHECK: llvm.call @wrap_random_normal_like({{.*}}) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32

    hip.random_normal_like(%ctx)
        ins(%input : memref<?x?x?xf16, 1>)
        outs(%output : memref<?x?x?xf16, 1>)
        {seed = 1.0 : f32}
    return
  }

  func.func @rnl_static_f32(
      %ctx: !hip.context,
      %input: memref<2x3xf32, 1>,
      %output: memref<2x3xf32, 1>) {
    // CHECK-LABEL: llvm.func @rnl_static_f32
    // CHECK: llvm.call @wrap_random_normal_like({{.*}}) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32

    hip.random_normal_like(%ctx)
        ins(%input : memref<2x3xf32, 1>)
        outs(%output : memref<2x3xf32, 1>)
        {mean = 2.0 : f32, scale = 0.5 : f32, seed = 3.0 : f32}
    return
  }
}
