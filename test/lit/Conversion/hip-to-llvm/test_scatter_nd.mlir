// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.scatter_nd (ONNX ScatterND opset 13) lowers to a wrap_scatter_nd
// runtime call.  Expected signature:
//   (state, data, indices, updates, output,
//    data_shape, data_rank, indices_shape, indices_rank,
//    data_type, indices_type, reduction)
//
// Covers:
//   - reduction=none on a 2-D f32 tensor with 3-row update slabs.
//   - reduction=add  on the same shapes (covers the atomic-add path).
//   - per-element scatter (indices innermost = data_rank, inner_block = 1).
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @scatter_nd_overwrite_2d_f32
  // CHECK: llvm.call @wrap_scatter_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64) -> i32
  func.func @scatter_nd_overwrite_2d_f32(%ctx: !hip.context,
                                         %data: memref<8x4xf32, 1>,
                                         %indices: memref<3x1xi64, 1>,
                                         %updates: memref<3x4xf32, 1>,
                                         %output: memref<8x4xf32, 1>) {
    hip.scatter_nd(%ctx) ins(%data, %indices, %updates :
        memref<8x4xf32, 1>, memref<3x1xi64, 1>, memref<3x4xf32, 1>)
        outs(%output : memref<8x4xf32, 1>)
        {reduction = 0 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @scatter_nd_add_2d_f32
  // CHECK: llvm.call @wrap_scatter_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64) -> i32
  func.func @scatter_nd_add_2d_f32(%ctx: !hip.context,
                                   %data: memref<8x4xf32, 1>,
                                   %indices: memref<3x1xi64, 1>,
                                   %updates: memref<3x4xf32, 1>,
                                   %output: memref<8x4xf32, 1>) {
    hip.scatter_nd(%ctx) ins(%data, %indices, %updates :
        memref<8x4xf32, 1>, memref<3x1xi64, 1>, memref<3x4xf32, 1>)
        outs(%output : memref<8x4xf32, 1>)
        {reduction = 1 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @scatter_nd_per_element_f16
  // CHECK: llvm.call @wrap_scatter_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64) -> i32
  func.func @scatter_nd_per_element_f16(%ctx: !hip.context,
                                        %data: memref<4x6xf16, 1>,
                                        %indices: memref<5x2xi32, 1>,
                                        %updates: memref<5xf16, 1>,
                                        %output: memref<4x6xf16, 1>) {
    hip.scatter_nd(%ctx) ins(%data, %indices, %updates :
        memref<4x6xf16, 1>, memref<5x2xi32, 1>, memref<5xf16, 1>)
        outs(%output : memref<4x6xf16, 1>)
        {reduction = 0 : i64}
    return
  }
}
