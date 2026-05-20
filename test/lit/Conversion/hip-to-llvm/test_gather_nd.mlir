// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.gather_nd lowers to llvm.call @wrap_gather_nd with signature
//   (state, data, indices, out,
//    data_shape_ptr, data_rank,
//    indices_shape_ptr, indices_rank,
//    out_shape_ptr, out_rank,
//    batch_dims, data_type) -> i32.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // No batch dims.
  func.func @gather_nd_default(
      %ctx: !hip.context,
      %data: memref<2x2xf32, 1>,
      %indices: memref<2x2xi64, 1>,
      %output: memref<2xf32, 1>) {
    // CHECK-LABEL: llvm.func @gather_nd_default

    hip.gather_nd(%ctx) ins(%data, %indices : memref<2x2xf32, 1>, memref<2x2xi64, 1>)
                        outs(%output : memref<2xf32, 1>)

    // Three shape arrays (data:2, indices:2, output:1).
    // CHECK: llvm.alloca {{.*}} x !llvm.array<2 x i64>
    // CHECK: llvm.alloca {{.*}} x !llvm.array<2 x i64>
    // CHECK: llvm.alloca {{.*}} x !llvm.array<1 x i64>
    // CHECK: llvm.call @wrap_gather_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
    return
  }

  // With batch_dims = 1.
  func.func @gather_nd_batched(
      %ctx: !hip.context,
      %data: memref<2x3x4xf32, 1>,
      %indices: memref<2x2x1xi64, 1>,
      %output: memref<2x2x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @gather_nd_batched

    hip.gather_nd(%ctx) ins(%data, %indices : memref<2x3x4xf32, 1>, memref<2x2x1xi64, 1>)
                        outs(%output : memref<2x2x4xf32, 1>)
                        {batch_dims = 1 : i64}

    // CHECK: llvm.call @wrap_gather_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
    return
  }

  // Dynamic indices.
  func.func @gather_nd_dynamic_indices(
      %ctx: !hip.context,
      %data: memref<2x2xf32, 1>,
      %indices: memref<?x2xi64, 1>,
      %output: memref<?xf32, 1>) {
    // CHECK-LABEL: llvm.func @gather_nd_dynamic_indices

    hip.gather_nd(%ctx) ins(%data, %indices : memref<2x2xf32, 1>, memref<?x2xi64, 1>)
                        outs(%output : memref<?xf32, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.call @wrap_gather_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
    return
  }
}
