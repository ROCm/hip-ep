// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify com.microsoft.MultiHeadAttention (via onnx.Custom) — the 8-input form
// used by Whisper's decoder cross-attention — lowers to hip.gqa with:
//   * no_causal = true (cross-attn is bidirectional: Q can see all K),
//   * num_heads == kv_num_heads (HPG=1; MHA, not group-query),
//   * a compile-time arith.constant seqlens_k = [Skv,…,Skv] (every cross-attn
//     batch element attends to the full encoder output, no padding),
//   * a compile-time arith.constant total_seq_len = Skv.
//
// Dispatch signal: 8-input MHA with K/V present AND past_key/past_value slots
// empty (operands[6,7] = NoValue) AND past_sequence_length empty
// (operands[8] missing/empty).  K/V come from encoder output (not a cache).
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Whisper-large-v3 decoder cross-attn: Q from decoder (1 token), K/V from
  // encoder output already split into BNSH (B=1, N=20 heads, Skv=1500, d=64).
  func.func @main_graph(
      %q:  tensor<1x1x1280xf16>,
      %k:  tensor<1x20x1500x64xf16>,
      %v:  tensor<1x20x1500x64xf16>)
      -> tensor<1x1x1280xf16> {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %none1 = "onnx.NoValue"() {value} : () -> none
    %none2 = "onnx.NoValue"() {value} : () -> none
    %none3 = "onnx.NoValue"() {value} : () -> none
    %none4 = "onnx.NoValue"() {value} : () -> none
    %none5 = "onnx.NoValue"() {value} : () -> none

    %0 = "onnx.Custom"(%q, %k, %v, %none1, %none2, %none3, %none4, %none5)
         <{function_name = "MultiHeadAttention"}>
         {domain_name = "com.microsoft",
          num_heads = 20 : si64,
          scale = 1.250000e-01 : f32,
          unidirectional = 0 : si64,
          mask_filter_value = -1.000000e+04 : f32}
         : (tensor<1x1x1280xf16>, tensor<1x20x1500x64xf16>, tensor<1x20x1500x64xf16>,
            none, none, none, none, none) -> tensor<1x1x1280xf16>

    // Constant seqlens_k = [1500] (1-element i32, batch=1) and
    // total_seq_len = 1500 (scalar i32). DAG because MLIR may print the
    // scalar before the 1-element tensor or vice versa.
    // CHECK-DAG: arith.constant dense<1500> : tensor<1xi32>
    // CHECK-DAG: arith.constant dense<1500> : tensor<i32>
    // The MHA op lowers to hip.gqa with no_causal=true, HPG=1.
    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: kv_num_heads = 20
    // CHECK-SAME: no_causal = true
    // CHECK-SAME: num_heads = 20
    // CHECK-NOT: hip.multi_head_attention

    return %0 : tensor<1x1x1280xf16>
  }

}
