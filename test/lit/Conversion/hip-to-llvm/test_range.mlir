// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @range_dynamic_i32(
      %ctx: !hip.context,
      %start: memref<i32, 1>,
      %limit: memref<i32, 1>,
      %delta: memref<i32, 1>,
      %output: memref<?xi32, 1>) {
    hip.range(%ctx) ins(%start, %limit, %delta : memref<i32, 1>, memref<i32, 1>, memref<i32, 1>)
                   outs(%output : memref<?xi32, 1>)
    return
  }

  func.func @range_static_f64(
      %ctx: !hip.context,
      %start: memref<f64, 1>,
      %limit: memref<f64, 1>,
      %delta: memref<f64, 1>,
      %output: memref<16xf64, 1>) {
    hip.range(%ctx) ins(%start, %limit, %delta : memref<f64, 1>, memref<f64, 1>, memref<f64, 1>)
                   outs(%output : memref<16xf64, 1>)
    return
  }

  func.func @range_dynamic_i16(
      %ctx: !hip.context,
      %start: memref<i16, 1>,
      %limit: memref<i16, 1>,
      %delta: memref<i16, 1>,
      %output: memref<?xi16, 1>) {
    hip.range(%ctx) ins(%start, %limit, %delta : memref<i16, 1>, memref<i16, 1>, memref<i16, 1>)
                   outs(%output : memref<?xi16, 1>)
    return
  }

  func.func @range_static_i64(
      %ctx: !hip.context,
      %start: memref<i64, 1>,
      %limit: memref<i64, 1>,
      %delta: memref<i64, 1>,
      %output: memref<16xi64, 1>) {
    hip.range(%ctx) ins(%start, %limit, %delta : memref<i64, 1>, memref<i64, 1>, memref<i64, 1>)
                   outs(%output : memref<16xi64, 1>)
    return
  }

  func.func @range_static_f32(
      %ctx: !hip.context,
      %start: memref<f32, 1>,
      %limit: memref<f32, 1>,
      %delta: memref<f32, 1>,
      %output: memref<16xf32, 1>) {
    hip.range(%ctx) ins(%start, %limit, %delta : memref<f32, 1>, memref<f32, 1>, memref<f32, 1>)
                   outs(%output : memref<16xf32, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @range_dynamic_i32
// CHECK: llvm.call @wrap_range({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

// CHECK-LABEL: llvm.func @range_static_f64
// CHECK: llvm.call @wrap_range({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

// CHECK-LABEL: llvm.func @range_dynamic_i16
// CHECK: llvm.call @wrap_range({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

// CHECK-LABEL: llvm.func @range_static_i64
// CHECK: llvm.call @wrap_range({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

// CHECK-LABEL: llvm.func @range_static_f32
// CHECK: llvm.call @wrap_range({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
