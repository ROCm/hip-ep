// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Document and lock-in the contract that ORT's GQA producer may pass a
// `seqlens_k` constant tensor of all `-1` to signal a fresh prefill (no past
// KV cache yet). The runtime fix in commit ecd443f makes
// `gqa_forward_hipblaslt` treat `seqlens_k_val < 0` as `past_len = 0` instead
// of rejecting it as an invalid index.
//
// This is an IR-level fixture only - the conversion pass does not interpret
// the contents of `seqlens_k`. The runtime branch is exercised by the
// MatMulNBits/GQA accuracy tests in CI; here we simply prove the conversion
// continues to lower the op cleanly when `seqlens_k` is a constant -1
// (encoded as i32 splat 0xFFFFFFFF).
//
// Companion runtime: lib/Runtime/real/gqa.cpp `if (seqlens_k_val < 0)`
// branch (around lines 305 and 418) - see the in-source comment that
// references this test.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===========================================================================
  // Test: GQA prefill with seqlens_k = constant -1 (fresh prefill sentinel)
  // ===========================================================================
  //
  // Shapes mirror Test 2 in test_gqa.mlir (packed-QKV prefill, sq=128) but
  // with `seqlens_k` materialised as a constant -1 tensor instead of being a
  // function argument. The lit test verifies the conversion still lowers
  // cleanly to hip.gqa; the runtime sentinel branch is what actually consumes
  // the value.
  // Function must be named @main_graph for --hip-add-context-arg to fire.
  func.func @main_graph(
      %query: tensor<1x128x12288xf16>,
      %past_key: tensor<1x8x0x128xf16>,
      %past_value: tensor<1x8x0x128xf16>,
      %total_seq_len: tensor<i32>)
      -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    // ORT prefill sentinel: seqlens_k[b] = -1 means "no past KV yet, this
    // is the first prefill step". onnx int32 splat -1 == dense<-1>.
    %seqlens_k = "onnx.Constant"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>

    %none_key = "onnx.NoValue"() {value} : () -> none
    %none_value = "onnx.NoValue"() {value} : () -> none
    %none1 = "onnx.NoValue"() {value} : () -> none
    %none2 = "onnx.NoValue"() {value} : () -> none

    %out:3 = "onnx.Custom"(%query, %none_key, %none_value, %past_key, %past_value,
                            %seqlens_k, %total_seq_len, %none1, %none2)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 32 : si64,
         kv_num_heads = 8 : si64,
         scale = 0.0883883461 : f32,
         softcap = 0.000000e+00 : f32,
         do_rotary = 0 : si64,
         rotary_interleaved = 0 : si64}
        : (tensor<1x128x12288xf16>, none, none,
           tensor<1x8x0x128xf16>, tensor<1x8x0x128xf16>,
           tensor<1xi32>, tensor<i32>, none, none)
        -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)

    // The constant -1 must reach the conversion intact and the op must
    // lower cleanly to hip.gqa. The onnx.Constant is lowered to a hip.constant
    // carrier (externalization is deferred to hip-externalize-constants); the
    // value must be preserved verbatim.
    // CHECK: hip.constant {value = dense<-1> : tensor<1xi32>}
    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: num_heads = 32
    // CHECK-NOT: onnx.Custom

    return %out#0, %out#1, %out#2 : tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
  }
}
