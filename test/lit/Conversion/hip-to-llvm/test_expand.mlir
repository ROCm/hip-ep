// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.expand lowers to llvm.call @wrap_expand with signature
//   (state, in, shape, out, in_shape_ptr, in_rank,
//    out_shape_ptr, out_rank, data_type) -> i32.
// Mirrors hip.tile in calling convention.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @expand_static_f32(
      %ctx: !hip.context,
      %x: memref<3x1xf32, 1>,
      %shape: memref<3xi64, 1>,
      %y: memref<2x3x6xf32, 1>) {
    // CHECK-LABEL: llvm.func @expand_static_f32

    hip.expand(%ctx) ins(%x, %shape : memref<3x1xf32, 1>, memref<3xi64, 1>)
                     outs(%y : memref<2x3x6xf32, 1>)

    // Input shape array (rank 2) + output shape array (rank 3).
    // CHECK: llvm.alloca {{.*}} x !llvm.array<2 x i64>
    // CHECK: llvm.alloca {{.*}} x !llvm.array<3 x i64>
    // CHECK: llvm.call @wrap_expand({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
    return
  }

  func.func @expand_dynamic(
      %ctx: !hip.context,
      %x: memref<3x1xf32, 1>,
      %shape: memref<3xi64, 1>,
      %y: memref<?x3x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @expand_dynamic

    hip.expand(%ctx) ins(%x, %shape : memref<3x1xf32, 1>, memref<3xi64, 1>)
                     outs(%y : memref<?x3x?xf32, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 2]
    // CHECK: llvm.call @wrap_expand({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
    return
  }

  // wrap_expand copies elements by width. A type with a runtime data type
  // keeps it (bf16 = 2); one without is passed as the integer type of the
  // same width (i1 as i8 = 5, ui32 as i32 = 3).
  func.func @expand_bf16(
      %ctx: !hip.context,
      %x: memref<1xbf16, 1>,
      %shape: memref<1xi64, 1>,
      %y: memref<?xbf16, 1>) {
    // CHECK-LABEL: llvm.func @expand_bf16

    hip.expand(%ctx) ins(%x, %shape : memref<1xbf16, 1>, memref<1xi64, 1>)
                     outs(%y : memref<?xbf16, 1>)

    // CHECK: %[[DT:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: llvm.call @wrap_expand(%{{.*}}, %[[DT]]) :
    return
  }

  func.func @expand_i1(
      %ctx: !hip.context,
      %x: memref<1xi1, 1>,
      %shape: memref<1xi64, 1>,
      %y: memref<?xi1, 1>) {
    // CHECK-LABEL: llvm.func @expand_i1

    hip.expand(%ctx) ins(%x, %shape : memref<1xi1, 1>, memref<1xi64, 1>)
                     outs(%y : memref<?xi1, 1>)

    // CHECK: %[[DT:.*]] = llvm.mlir.constant(5 : i64) : i64
    // CHECK: llvm.call @wrap_expand(%{{.*}}, %[[DT]]) :
    return
  }

  func.func @expand_ui32(
      %ctx: !hip.context,
      %x: memref<1xui32, 1>,
      %shape: memref<1xi64, 1>,
      %y: memref<?xui32, 1>) {
    // CHECK-LABEL: llvm.func @expand_ui32

    hip.expand(%ctx) ins(%x, %shape : memref<1xui32, 1>, memref<1xi64, 1>)
                     outs(%y : memref<?xui32, 1>)

    // CHECK: %[[DT:.*]] = llvm.mlir.constant(3 : i64) : i64
    // CHECK: llvm.call @wrap_expand(%{{.*}}, %[[DT]]) :
    return
  }
}
