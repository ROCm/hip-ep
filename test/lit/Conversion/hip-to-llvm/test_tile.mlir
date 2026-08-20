// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.tile lowers to llvm.call @wrap_tile with signature
//   (state, in, repeats, out, in_shape_ptr, in_rank,
//    out_shape_ptr, out_rank, data_type) -> i32.
// Both in_shape and out_shape are stack-allocated arrays of i64 populated
// via GEP + store of each dim size (static const or dynamic
// extractvalue).
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @checked_tile_extent(
      %ctx: !hip.context, %valid: i1, %input: index,
      %repeat: i64, %prior: index) -> index {
    %nextValid, %extent, %elements = hip.checked_tile_extent(
      %ctx, %valid, %input, %repeat, %prior)
      expected_extent = -1 -> (i1, index, index)
    return %extent : index
  }

  // CHECK-LABEL: llvm.func @checked_tile_extent
  // CHECK-COUNT-2: llvm.store
  // CHECK: %[[STATUS:.*]] = llvm.call @hipdnn_ep_checked_tile_extent
  // CHECK: llvm.icmp "eq" %[[STATUS]]
  // CHECK: llvm.select

  func.func @tile_static_f32(
      %ctx: !hip.context,
      %x: memref<2x3xf32, 1>,
      %repeats: memref<2xi64, 1>,
      %y: memref<4x9xf32, 1>) {
    // CHECK-LABEL: llvm.func @tile_static_f32

    hip.tile(%ctx) ins(%x, %repeats : memref<2x3xf32, 1>, memref<2xi64, 1>)
                   outs(%y : memref<4x9xf32, 1>)

    // Two shape-arrays (one per memref) of length 2.
    // CHECK: llvm.alloca {{.*}} x !llvm.array<2 x i64>
    // CHECK: llvm.alloca {{.*}} x !llvm.array<2 x i64>
    // CHECK: llvm.call @wrap_tile({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
    return
  }

  func.func @tile_dynamic(
      %ctx: !hip.context,
      %x: memref<?x?xf32, 1>,
      %repeats: memref<2xi64, 1>,
      %y: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @tile_dynamic

    hip.tile(%ctx) ins(%x, %repeats : memref<?x?xf32, 1>, memref<2xi64, 1>)
                   outs(%y : memref<?x?xf32, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.call @wrap_tile({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
    return
  }
}
