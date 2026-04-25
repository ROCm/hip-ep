// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.stft is lowered to llvm.call @wrap_stft with the expected
// argument list (state, signal, window, output, batch, signal_len,
// frame_step, frame_length, n_frames, n_freqs, onesided, data_type).
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // ---- With explicit window ----
  func.func @stft_with_window(
      %ctx: !hip.context,
      %signal: memref<1x320xf32, 1>,
      %window: memref<20xf32, 1>,
      %output: memref<1x76x11x2xf32, 1>) {
    // CHECK-LABEL: llvm.func @stft_with_window

    hip.stft(%ctx) ins(%signal, %window
                       : memref<1x320xf32, 1>, memref<20xf32, 1>)
                   outs(%output : memref<1x76x11x2xf32, 1>)
                   {frame_step = 4 : i64, frame_length = 20 : i64,
                    onesided = 1 : i64}

    // CHECK: llvm.call @wrap_stft({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // ---- Window absent: rectangular window, runtime gets a null pointer ----
  func.func @stft_no_window(
      %ctx: !hip.context,
      %signal: memref<1x320xf32, 1>,
      %output: memref<1x76x11x2xf32, 1>) {
    // CHECK-LABEL: llvm.func @stft_no_window

    hip.stft(%ctx) ins(%signal : memref<1x320xf32, 1>)
                   outs(%output : memref<1x76x11x2xf32, 1>)
                   {frame_step = 4 : i64, frame_length = 20 : i64,
                    onesided = 1 : i64}

    // CHECK: llvm.mlir.zero
    // CHECK: llvm.call @wrap_stft({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
