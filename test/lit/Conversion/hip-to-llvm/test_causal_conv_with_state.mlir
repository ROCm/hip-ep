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
// - Attributes (activation, ndim, channels_last) converted to integer args
// - Shape info (batch, channels, seq_len, kernel_size) extracted:
//     * static dims become llvm.mlir.constant
//     * dynamic dims become llvm.extractvalue %desc[3, N] (+ llvm.mul for
//       composite sizes like seq_len and kernel_size)
//     * channels_last reads channels from dim 2 and seq_len from dim 1
//
// Expected: wrap_causal_conv_with_state(state, input, weight, bias,
//           past_state, output, present_state, batch_size, channels,
//           seq_len, kernel_size, ndim, activation, element_size_bytes,
//           channels_last)
//
// Test cases:
// 1. causal_conv_full                  — all static, bias + past_state, silu
// 2. causal_conv_no_optional           — all static, no bias / past_state
// 3. causal_conv_no_activation         — all static, activation=none
// 4. causal_conv_dynamic_activations   — dynamic input/output, static weight
// 5. causal_conv_fully_dynamic         — dynamic input AND dynamic weight
//                                         (validates kernel_size extraction)
// 6. causal_conv_channels_last         — (B, L, C) input, static
// 7. causal_conv_channels_last_dynamic — (B, L, C) input, dynamic L
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --convert-hip-to-llvm | FileCheck %s
//
// The trailing i32 on each wrap_causal_conv_with_state call is op_state_slot,
// threaded by --assign-op-state-slots. It selects the per-instance
// CausalConvState (descriptor/algo cache), replacing the former shared
// RuntimeState::causal_conv_cache.

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

    // CHECK: llvm.call @wrap_causal_conv_with_state({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

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

    // CHECK: llvm.call @wrap_causal_conv_with_state({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

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

    // CHECK: llvm.call @wrap_causal_conv_with_state({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // --------------------------------------------------------------------------
  // Test 4: Dynamic input/output (batch + seq_len dynamic, weight static).
  // Typical LLM incremental-decode scenario.
  //
  // Expectations:
  // - batch_size / channels / seq_len extracted at runtime via
  //   llvm.extractvalue from the input MemRef descriptor (indices [3, 0..2]).
  // - kernel_size still a compile-time constant since weight is static.
  // --------------------------------------------------------------------------
  func.func @causal_conv_dynamic_activations(
      %ctx: !hip.context,
      %input: memref<?x?x?xf16, 1>,
      %weight: memref<64x1x4xf16, 1>,
      %bias: memref<64xf16, 1>,
      %past_state: memref<?x?x3xf16, 1>,
      %output: memref<?x?x?xf16, 1>,
      %present_state: memref<?x?x3xf16, 1>) {
    // CHECK-LABEL: llvm.func @causal_conv_dynamic_activations

    hip.causal_conv_with_state(%ctx)
        ins(%input, %weight, %bias, %past_state :
            memref<?x?x?xf16, 1>, memref<64x1x4xf16, 1>,
            memref<64xf16, 1>, memref<?x?x3xf16, 1>)
        outs(%output, %present_state :
             memref<?x?x?xf16, 1>, memref<?x?x3xf16, 1>)
        {activation = "silu", ndim = 1 : i64}

    // Runtime-extracted dims from input descriptor: batch, channels, seq_len.
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // CHECK: llvm.extractvalue {{.*}}[3, 2]
    // kernel_size = 4 remains a compile-time constant from static weight.
    // CHECK: llvm.mlir.constant(4 : i64)
    // CHECK: llvm.call @wrap_causal_conv_with_state({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // --------------------------------------------------------------------------
  // Test 5: Fully dynamic (input, weight, and state all dynamic).
  // Validates the kernel_size fix: weight's kernel dim must also be extracted
  // at runtime via llvm.extractvalue, not multiplied as kDynamic sentinel.
  // --------------------------------------------------------------------------
  func.func @causal_conv_fully_dynamic(
      %ctx: !hip.context,
      %input: memref<?x?x?xf16, 1>,
      %weight: memref<?x1x?xf16, 1>,
      %output: memref<?x?x?xf16, 1>,
      %present_state: memref<?x?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @causal_conv_fully_dynamic

    hip.causal_conv_with_state(%ctx)
        ins(%input, %weight :
            memref<?x?x?xf16, 1>, memref<?x1x?xf16, 1>)
        outs(%output, %present_state :
             memref<?x?x?xf16, 1>, memref<?x?x?xf16, 1>)
        {ndim = 1 : i64}

    // All 3 input dims extracted at runtime.
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // CHECK: llvm.extractvalue {{.*}}[3, 2]
    // kernel_size is now built as: 1 * weight.dim[2] at runtime.
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.extractvalue {{.*}}[3, 2]
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_causal_conv_with_state({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // --------------------------------------------------------------------------
  // Test 6: channels_last. Input is (batch, seq_len, channels), so channels
  // comes from dim 2 and seq_len from dim 1 -- the opposite of every case
  // above. The two extents are deliberately different (L=128, C=64) so the
  // emission order below distinguishes a correct swap from a no-op: the
  // arguments are built batch, channels, seq_len, so the constants must appear
  // as 2, 64, 128. Reading the dims channels-first would emit 2, 128, 64 and
  // fail here rather than silently passing a transposed shape to the kernel.
  // --------------------------------------------------------------------------
  func.func @causal_conv_channels_last(
      %ctx: !hip.context,
      %input: memref<2x128x64xf16, 1>,
      %weight: memref<64x1x4xf16, 1>,
      %bias: memref<64xf16, 1>,
      %past_state: memref<2x64x3xf16, 1>,
      %output: memref<2x128x64xf16, 1>,
      %present_state: memref<2x64x3xf16, 1>) {
    // CHECK-LABEL: llvm.func @causal_conv_channels_last

    hip.causal_conv_with_state(%ctx)
        ins(%input, %weight, %bias, %past_state :
            memref<2x128x64xf16, 1>, memref<64x1x4xf16, 1>,
            memref<64xf16, 1>, memref<2x64x3xf16, 1>)
        outs(%output, %present_state :
             memref<2x128x64xf16, 1>, memref<2x64x3xf16, 1>)
        {activation = "silu", ndim = 1 : i64, channels_last = true}

    // batch = 2, then channels = 64 (dim 2), then seq_len = 128 (dim 1).
    // CHECK: llvm.mlir.constant(2 : i64)
    // CHECK: llvm.mlir.constant(64 : i64)
    // CHECK: llvm.mlir.constant(128 : i64)
    // CHECK: llvm.call @wrap_causal_conv_with_state({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // --------------------------------------------------------------------------
  // Test 7: channels_last with a dynamic sequence length, which is the shape
  // the model actually presents (prompt length varies, channels does not).
  // seq_len is extracted from descriptor index [3, 1] and channels stays a
  // compile-time constant from dim 2.
  // --------------------------------------------------------------------------
  func.func @causal_conv_channels_last_dynamic(
      %ctx: !hip.context,
      %input: memref<1x?x64xf16, 1>,
      %weight: memref<64x1x4xf16, 1>,
      %output: memref<1x?x64xf16, 1>,
      %present_state: memref<1x64x3xf16, 1>) {
    // CHECK-LABEL: llvm.func @causal_conv_channels_last_dynamic

    hip.causal_conv_with_state(%ctx)
        ins(%input, %weight :
            memref<1x?x64xf16, 1>, memref<64x1x4xf16, 1>)
        outs(%output, %present_state :
             memref<1x?x64xf16, 1>, memref<1x64x3xf16, 1>)
        {activation = "silu", ndim = 1 : i64, channels_last = true}

    // channels is static (dim 2); only seq_len (dim 1) is extracted.
    // CHECK: llvm.mlir.constant(64 : i64)
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // CHECK: llvm.call @wrap_causal_conv_with_state({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
