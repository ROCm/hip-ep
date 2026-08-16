// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify com.microsoft.MultiHeadAttention (via onnx.Custom) — the 9-input form
// produced by Task-9 ONNX surgery (past_sequence_length input injected into
// slot 8) used by Whisper's decoder self-attention — lowers to hip.gqa with:
//   * no_causal = false (decoder self-attn IS causal / autoregressive),
//   * num_heads == kv_num_heads (HPG=1; MHA, not group-query),
//   * seqlens_k = operands[8]  (the runtime past_sequence_length tensor),
//   * total_seq_len = arith.constant <static cache buffer length>  (taken
//     from the present_key BNSH shape's dim 2 — baked in at compile time).
//
// Dispatch signal: 9-input MHA with K/V present, past_key/past_value present,
// the past_sequence_length operand (slot 8) present, AND no extra
// bias/mask/attention_bias/cache_indirection inputs. The presence of the
// past_sequence_length OPERAND is the discriminator — NOT a
// past_present_share_buffer attribute (ORT's MHA schema rejects that
// attribute, so the surgery threads only the input).
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Whisper-large-v3 decoder self-attn after Task-9 surgery: Q/K/V from the
  // current step, past KV (BNSH, B=1, N=20, max_kv=448, d=64), and an extra
  // past_sequence_length (1-D i32) operand telling the runtime how many KV
  // slots are already populated.
  func.func @main_graph(
      %q:        tensor<1x1x1280xf16>,
      %k:        tensor<1x1x1280xf16>,
      %v:        tensor<1x1x1280xf16>,
      %past_k:   tensor<1x20x448x64xf16>,
      %past_v:   tensor<1x20x448x64xf16>,
      %past_seq: tensor<1xi32>)
      -> (tensor<1x1x1280xf16>, tensor<1x20x448x64xf16>, tensor<1x20x448x64xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,
    // Capture the past_seq SSA name (last function argument here) so we can
    // assert hip.gqa consumes it as seqlens_k.
    // CHECK-SAME: %[[PAST_SEQ:[a-zA-Z0-9_]+]]: tensor<1xi32>

    %none1 = "onnx.NoValue"() {value} : () -> none
    %none2 = "onnx.NoValue"() {value} : () -> none
    %none3 = "onnx.NoValue"() {value} : () -> none

    %out:3 = "onnx.Custom"(%q, %k, %v, %none1, %none2, %none3, %past_k, %past_v, %past_seq)
        <{function_name = "MultiHeadAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 20 : si64,
         scale = 1.250000e-01 : f32,
         unidirectional = 1 : si64,
         mask_filter_value = -1.000000e+04 : f32}
        : (tensor<1x1x1280xf16>, tensor<1x1x1280xf16>, tensor<1x1x1280xf16>,
           none, none, none, tensor<1x20x448x64xf16>, tensor<1x20x448x64xf16>,
           tensor<1xi32>)
        -> (tensor<1x1x1280xf16>, tensor<1x20x448x64xf16>, tensor<1x20x448x64xf16>)

    // Compile-time total_seq_len constant = cache buffer length (448).
    // CHECK: arith.constant dense<448> : tensor<i32>
    // hip.gqa with no_causal=false (default → omitted by printer), HPG=1,
    // consumes the runtime past_sequence_length as seqlens_k (asserted via
    // %[[PAST_SEQ]] appearing in the ins() list).
    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: %[[PAST_SEQ]]
    // CHECK-SAME: kv_num_heads = 20
    // CHECK-SAME: num_heads = 20
    // CHECK-NOT: no_causal
    // CHECK-NOT: hip.multi_head_attention

    return %out#0, %out#1, %out#2
      : tensor<1x1x1280xf16>, tensor<1x20x448x64xf16>, tensor<1x20x448x64xf16>
  }

}
