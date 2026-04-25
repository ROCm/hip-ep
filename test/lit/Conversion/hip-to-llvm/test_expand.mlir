// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.expand (ONNX Expand: numpy-style broadcast) lowers to a
// wrap_expand runtime call with the expected
//   (state, in_ptr, out_ptr, in_shape, in_strides, out_shape,
//    in_rank, out_rank, data_type)
// signature.
//
// Covers:
//   - Same-rank trailing-axis broadcast (1x256x1 -> 1x256x128).
//   - Rank-prepending broadcast (256x1 -> 1x256x128) -- exercises the
//     right-aligning logic in the runtime wrapper.
//   - Mixed broadcast within the same rank for fp16 (1x4x1 -> 1x4x8).
//   - Int64 expand (used by Kokoro shape ops).
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @expand_trailing_axis_f32
  // CHECK: llvm.call @wrap_expand({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  func.func @expand_trailing_axis_f32(%ctx: !hip.context,
                                      %input: memref<1x256x1xf32, 1>,
                                      %output: memref<1x256x128xf32, 1>) {
    hip.expand(%ctx) ins(%input : memref<1x256x1xf32, 1>)
                     outs(%output : memref<1x256x128xf32, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @expand_rank_prepend_f32
  // CHECK: llvm.call @wrap_expand({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  func.func @expand_rank_prepend_f32(%ctx: !hip.context,
                                     %input: memref<256x1xf32, 1>,
                                     %output: memref<1x256x128xf32, 1>) {
    hip.expand(%ctx) ins(%input : memref<256x1xf32, 1>)
                     outs(%output : memref<1x256x128xf32, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @expand_mixed_axes_f16
  // CHECK: llvm.call @wrap_expand({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  func.func @expand_mixed_axes_f16(%ctx: !hip.context,
                                   %input: memref<1x4x1xf16, 1>,
                                   %output: memref<1x4x8xf16, 1>) {
    hip.expand(%ctx) ins(%input : memref<1x4x1xf16, 1>)
                     outs(%output : memref<1x4x8xf16, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @expand_i64
  // CHECK: llvm.call @wrap_expand({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  func.func @expand_i64(%ctx: !hip.context,
                        %input: memref<1xi64, 1>,
                        %output: memref<8xi64, 1>) {
    hip.expand(%ctx) ins(%input : memref<1xi64, 1>)
                     outs(%output : memref<8xi64, 1>)
    return
  }
}
