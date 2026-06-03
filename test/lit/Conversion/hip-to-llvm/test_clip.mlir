// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @clip_static_f32(
      %ctx: !hip.context,
      %input: memref<3x4xf32, 1>,
      %lo: memref<f32, 1>,
      %hi: memref<f32, 1>,
      %output: memref<3x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @clip_static_f32
    hip.clip(%ctx) ins(%input, %lo, %hi : memref<3x4xf32, 1>,
                                          memref<f32, 1>,
                                          memref<f32, 1>)
                   outs(%output : memref<3x4xf32, 1>)
    // CHECK: llvm.call @wrap_clip
    return
  }

  func.func @clip_dynamic_f16(
      %ctx: !hip.context,
      %input: memref<?x?xf16, 1>,
      %lo: memref<f16, 1>,
      %hi: memref<f16, 1>,
      %output: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @clip_dynamic_f16
    hip.clip(%ctx) ins(%input, %lo, %hi : memref<?x?xf16, 1>,
                                          memref<f16, 1>,
                                          memref<f16, 1>)
                   outs(%output : memref<?x?xf16, 1>)
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.mul
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_clip
    return
  }
}
