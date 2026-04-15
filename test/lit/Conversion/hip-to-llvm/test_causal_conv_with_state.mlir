// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP causal_conv_with_state operation is correctly lowered to LLVM
// call to wrap_causal_conv_with_state runtime function.
//
// This test validates:
// - hip.causal_conv_with_state -> llvm.call @wrap_causal_conv_with_state
// - Type conversion: !hip.context -> !llvm.ptr
// - Optional inputs (bias, past_state) passed as pointers (null when absent)
// - Two output buffers (output, present_state) forwarded
// - Attributes (activation, ndim) converted to integer args
// - Shape info (batch, channels, seq_len, kernel_size) extracted
//
// Expected: wrap_causal_conv_with_state(state, input, weight, bias,
//           past_state, output, present_state, batch_size, channels,
//           seq_len, kernel_size, ndim, activation, element_size_bytes)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Full op with bias and past_state, activation=silu
  func.func @causal_conv_full(
      %ctx: !hip.context,
      %input: memref<1x64x128xf16, 1>,
      %weight: memref<64x1x4xf16, 1>,
      %bias: memref<64xf16, 1>,
      %past_state: memref<1x64x3xf16, 1>,
      %output: memref<1x64x128xf16, 1>,
      %present_state: memref<1x64x3xf16, 1>) {
    // CHECK-LABEL: llvm.func @causal_conv_full
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr

    hip.causal_conv_with_state(%ctx)
        ins(%input, %weight, %bias, %past_state :
            memref<1x64x128xf16, 1>, memref<64x1x4xf16, 1>,
            memref<64xf16, 1>, memref<1x64x3xf16, 1>)
        outs(%output, %present_state :
             memref<1x64x128xf16, 1>, memref<1x64x3xf16, 1>)
        {activation = "silu", ndim = 1 : i64}

    // CHECK: llvm.call @wrap_causal_conv_with_state({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 2: No optional inputs (no bias, no past_state)
  func.func @causal_conv_no_optional(
      %ctx: !hip.context,
      %input: memref<1x64x128xf16, 1>,
      %weight: memref<64x1x4xf16, 1>,
      %output: memref<1x64x128xf16, 1>,
      %present_state: memref<1x64x3xf16, 1>) {
    // CHECK-LABEL: llvm.func @causal_conv_no_optional

    hip.causal_conv_with_state(%ctx)
        ins(%input, %weight :
            memref<1x64x128xf16, 1>, memref<64x1x4xf16, 1>)
        outs(%output, %present_state :
             memref<1x64x128xf16, 1>, memref<1x64x3xf16, 1>)
        {ndim = 1 : i64}

    // CHECK: llvm.call @wrap_causal_conv_with_state({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 3: Different shape, activation=none
  func.func @causal_conv_no_activation(
      %ctx: !hip.context,
      %input: memref<2x128x64xf16, 1>,
      %weight: memref<128x1x3xf16, 1>,
      %bias: memref<128xf16, 1>,
      %past_state: memref<2x128x2xf16, 1>,
      %output: memref<2x128x64xf16, 1>,
      %present_state: memref<2x128x2xf16, 1>) {
    // CHECK-LABEL: llvm.func @causal_conv_no_activation

    hip.causal_conv_with_state(%ctx)
        ins(%input, %weight, %bias, %past_state :
            memref<2x128x64xf16, 1>, memref<128x1x3xf16, 1>,
            memref<128xf16, 1>, memref<2x128x2xf16, 1>)
        outs(%output, %present_state :
             memref<2x128x64xf16, 1>, memref<2x128x2xf16, 1>)
        {activation = "none", ndim = 1 : i64}

    // CHECK: llvm.call @wrap_causal_conv_with_state({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
