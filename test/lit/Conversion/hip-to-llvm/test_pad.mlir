// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.pad is lowered to llvm.call @wrap_pad with the expected
// argument list (state, in_ptr, out_ptr, in_shape, in_strides,
// out_shape, out_strides, rank, pads_begin, pads_begin_len,
// data_type, mode, value).
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Reflect on innermost axis of an NCW tensor (Kokoro shape).
  func.func @pad_reflect_innermost_f16(
      %ctx: !hip.context,
      %input: memref<1x1x16xf16, 1>,
      %output: memref<1x1x18xf16, 1>) {
    // CHECK-LABEL: llvm.func @pad_reflect_innermost_f16

    hip.pad(%ctx) ins(%input : memref<1x1x16xf16, 1>)
                  outs(%output : memref<1x1x18xf16, 1>)
                  {pads_begin = [0, 0, 1],
                   pads_end = [0, 0, 1],
                   mode = 1 : i64,
                   value = 0.0 : f32}

    // CHECK: llvm.call @wrap_pad({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, f32) -> i32

    return
  }

  // Constant pad on a rank-2 f32 tensor.
  func.func @pad_constant_f32(
      %ctx: !hip.context,
      %input: memref<4x6xf32, 1>,
      %output: memref<6x8xf32, 1>) {
    // CHECK-LABEL: llvm.func @pad_constant_f32

    hip.pad(%ctx) ins(%input : memref<4x6xf32, 1>)
                  outs(%output : memref<6x8xf32, 1>)
                  {pads_begin = [1, 1],
                   pads_end = [1, 1],
                   mode = 0 : i64,
                   value = -1.5 : f32}

    // CHECK: llvm.call @wrap_pad({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, f32) -> i32

    return
  }

  // Edge mode on bf16.
  func.func @pad_edge_bf16(
      %ctx: !hip.context,
      %input: memref<2x3xbf16, 1>,
      %output: memref<2x5xbf16, 1>) {
    // CHECK-LABEL: llvm.func @pad_edge_bf16

    hip.pad(%ctx) ins(%input : memref<2x3xbf16, 1>)
                  outs(%output : memref<2x5xbf16, 1>)
                  {pads_begin = [0, 1],
                   pads_end = [0, 1],
                   mode = 2 : i64,
                   value = 0.0 : f32}

    // CHECK: llvm.call @wrap_pad({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, f32) -> i32

    return
  }
}
