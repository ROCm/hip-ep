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
}
