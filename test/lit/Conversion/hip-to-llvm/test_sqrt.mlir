// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK: llvm.func @wrap_power(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64, f64, f64) -> i32

module {
  // Test 1: Static 1D f32
  func.func @sqrt_static_1d_f32(
      %ctx: !hip.context,
      %input: memref<128xf32, 1>,
      %output: memref<128xf32, 1>) {
    // CHECK-LABEL: llvm.func @sqrt_static_1d_f32

    hip.sqrt(%ctx) ins(%input : memref<128xf32, 1>)
                   outs(%output : memref<128xf32, 1>)

    // CHECK: %{{.*}} = llvm.mlir.constant(128 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
    // CHECK: llvm.call @wrap_power({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64, f64, f64) -> i32

    return
  }

  // Test 2: Static 2D f16
  func.func @sqrt_static_2d_f16(
      %ctx: !hip.context,
      %input: memref<128x512xf16, 1>,
      %output: memref<128x512xf16, 1>) {
    // CHECK-LABEL: llvm.func @sqrt_static_2d_f16

    hip.sqrt(%ctx) ins(%input : memref<128x512xf16, 1>)
                   outs(%output : memref<128x512xf16, 1>)

    // CHECK: llvm.mlir.constant(128 : i64)
    // CHECK: llvm.mlir.constant(512 : i64)
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @wrap_power({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64, f64, f64) -> i32

    return
  }

  // Test 3: Dynamic 2D f32
  func.func @sqrt_dynamic_2d_f32(
      %ctx: !hip.context,
      %input: memref<?x?xf32, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @sqrt_dynamic_2d_f32

    hip.sqrt(%ctx) ins(%input : memref<?x?xf32, 1>)
                   outs(%output : memref<?x?xf32, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.mlir.constant(0 : i64)
    // CHECK: llvm.call @wrap_power({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64, f64, f64) -> i32

    return
  }
}
