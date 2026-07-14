// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify native onnx.Attention (ONNX opset 23/24) lowers to hip.gqa with:
//   - external attn_mask threaded as attention_bias
//   - is_causal=0 -> no_causal=true (built-in causal mask skipped)
//   - present KV seq extent = past_seq + current_seq (concat semantics),
//     and seqlens_k = total_seq - 1 (ORT convention total_seq = seqlens_k + 1)
//   - present_key/value DPS inits sized to the past+current total
//
// Two cases: a static-shape decode (16 query heads, 8 KV heads), and a
// dynamic-shape prefill with an EMPTY past (the case that previously sized
// present from dim(past_key, 2) alone -> zero-length present buffer ->
// null present_key -> zeroed attention output).
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @gemma_decode_attention(
      %query: tensor<1x1x2048xf16>,
      %key: tensor<1x1x1024xf16>,
      %value: tensor<1x1x1024xf16>,
      %attn_mask: tensor<1x1x1x128xf16>,
      %past_key: tensor<1x8x127x128xf16>,
      %past_value: tensor<1x8x127x128xf16>)
      -> (tensor<1x1x2048xf16>, tensor<1x8x128x128xf16>,
          tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @gemma_decode_attention
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %out:3 = "onnx.Attention"(%query, %key, %value, %attn_mask, %past_key,
                              %past_value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 8 : si64,
         is_causal = 0 : si64,
         scale = 0.0883883461 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<1x1x2048xf16>, tensor<1x1x1024xf16>, tensor<1x1x1024xf16>,
           tensor<1x1x1x128xf16>, tensor<1x8x127x128xf16>,
           tensor<1x8x127x128xf16>)
        -> (tensor<1x1x2048xf16>, tensor<1x8x128x128xf16>,
            tensor<1x8x128x128xf16>)

    // present KV seq = past_seq (127) + current_seq (1); seqlens_k = total - 1.
    // CHECK: tensor.dim %{{.*}}, %c2
    // CHECK: arith.addi
    // CHECK: arith.subi
    // CHECK: tensor.from_elements %{{.*}} : tensor<1xi32>
    // CHECK: tensor.empty() : tensor<1x1x2048xf16>
    // CHECK: tensor.empty() : tensor<1x8x128x128xf16>
    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK-SAME: num_heads = 16
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: no_causal = true
    // CHECK-NOT: onnx.Attention

    return %out#0, %out#1, %out#2
        : tensor<1x1x2048xf16>, tensor<1x8x128x128xf16>,
          tensor<1x8x128x128xf16>
  }

  // Dynamic-shape prefill with an EMPTY past: present seq is computed at
  // runtime as past_seq + current_seq (NOT dim(past_key, 2) alone), so the
  // present buffer is sized to the real KV length even when past_seq == 0.
  // Mirrors the Gemma-4 decoder self-attn (16 query heads, 8 KV heads, d=256).
  func.func @gemma_prefill_attention_dynamic(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x2048xf16>,
      %value: tensor<?x?x2048xf16>,
      %attn_mask: tensor<?x1x?x?xf16>,
      %past_key: tensor<?x8x?x256xf16>,
      %past_value: tensor<?x8x?x256xf16>)
      -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
          tensor<?x8x?x256xf16>) {

    // CHECK-LABEL: func.func @gemma_prefill_attention_dynamic
    // CHECK-SAME: (%[[CTX2:.*]]: !hip.context,

    %out:3 = "onnx.Attention"(%query, %key, %value, %attn_mask, %past_key,
                              %past_value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 8 : si64,
         is_causal = 0 : si64,
         scale = 1.000000e+00 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<?x?x4096xf16>, tensor<?x?x2048xf16>, tensor<?x?x2048xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x256xf16>,
           tensor<?x8x?x256xf16>)
        -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
            tensor<?x8x?x256xf16>)

    // present seq extent = past + current, materialised as arith.addi and fed
    // into the present tensor.empty (dynamic seq dim), NOT dim(past_key, 2).
    // CHECK: %[[CUR:.*]] = tensor.dim %{{.*}}, %c1
    // CHECK: %[[PAST:.*]] = tensor.dim %{{.*}}, %c2
    // CHECK: %[[TOT:.*]] = arith.addi %[[PAST]], %[[CUR]]
    // CHECK: arith.subi %[[TOT]], %{{.*}}
    // CHECK: tensor.from_elements %{{.*}} : tensor<1xi32>
    // CHECK: tensor.empty(%{{.*}}, %[[TOT]]) : tensor<?x8x?x256xf16>
    // CHECK: hip.gqa(%[[CTX2]])
    // CHECK-SAME: no_causal = true
    // CHECK-NOT: onnx.Attention

    return %out#0, %out#1, %out#2
        : tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
          tensor<?x8x?x256xf16>
  }
}
