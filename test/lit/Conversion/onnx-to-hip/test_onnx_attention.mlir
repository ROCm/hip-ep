// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify native onnx.Attention (ONNX opset 23/24) lowers to hip.gqa with:
//   - external attn_mask threaded as attention_bias
//   - is_causal=0 -> no_causal=true (built-in causal mask skipped)
//   - is_causal=1 WITH an external attn_mask -> no_causal=false AND the mask
//     threaded as attention_bias (runtime adds the mask, then the built-in
//     causal triangle applies on top -- both, mirroring the ONNX reference)
//   - present KV seq extent = past_seq + current_seq (concat semantics),
//     and seqlens_k = total_seq - 1 (ORT convention total_seq = seqlens_k + 1)
//   - present_key/value DPS inits sized to the past+current total
//
// Cases: a static-shape decode (16 query heads, 8 KV heads), a dynamic-shape
// prefill with an EMPTY past (the case that previously sized present from
// dim(past_key, 2) alone -> zero-length present buffer -> null present_key ->
// zeroed attention output), the Gemma-4-E2B decoder self-attn (is_causal=1 +
// fp16 mask + past KV + 3 outputs -- previously rejected), a single-output
// causal case, a bidirectional no-mask case, and a rank-4 BNSH case.
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

  // Gemma-4-E2B decoder self-attn: is_causal=1 WITH an external fp16 additive
  // mask, past KV cache, 3 outputs, rank-3 Q/K/V (q_num_heads=8, kv_num_heads=1,
  // head_dim=256). This exact combination (causal + explicit mask) was
  // previously REJECTED by the converter, leaving onnx.Attention unconverted ->
  // "op was not bufferized". Now it converts: no_causal=false (built-in causal
  // still applies) AND the mask is threaded as attention_bias (added first).
  func.func @gemma_decoder_causal_with_mask(
      %query: tensor<?x?x2048xf16>,
      %key: tensor<?x?x256xf16>,
      %value: tensor<?x?x256xf16>,
      %attn_mask: tensor<?x1x?x?xf16>,
      %past_key: tensor<?x1x?x256xf16>,
      %past_value: tensor<?x1x?x256xf16>)
      -> (tensor<?x?x2048xf16>, tensor<?x1x?x256xf16>,
          tensor<?x1x?x256xf16>) {

    // CHECK-LABEL: func.func @gemma_decoder_causal_with_mask
    // CHECK-SAME: (%[[CTXG:.*]]: !hip.context,

    %out:3 = "onnx.Attention"(%query, %key, %value, %attn_mask, %past_key,
                              %past_value)
        {q_num_heads = 8 : si64,
         kv_num_heads = 1 : si64,
         is_causal = 1 : si64,
         scale = 1.000000e+00 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<?x?x2048xf16>, tensor<?x?x256xf16>, tensor<?x?x256xf16>,
           tensor<?x1x?x?xf16>, tensor<?x1x?x256xf16>,
           tensor<?x1x?x256xf16>)
        -> (tensor<?x?x2048xf16>, tensor<?x1x?x256xf16>,
            tensor<?x1x?x256xf16>)

    // present seq extent = past + current; mask threaded as attention_bias (8
    // hip.gqa ins: q, k, v, past_k, past_v, seqlens_k, total_seq, mask).
    // CHECK: %[[CURG:.*]] = tensor.dim %{{.*}}, %c1
    // CHECK: %[[PASTG:.*]] = tensor.dim %{{.*}}, %c2
    // CHECK: %[[TOTG:.*]] = arith.addi %[[PASTG]], %[[CURG]]
    // CHECK: tensor.from_elements %{{.*}} : tensor<1xi32>
    // CHECK: tensor.empty(%{{.*}}, %[[TOTG]]) : tensor<?x1x?x256xf16>
    // CHECK: hip.gqa(%[[CTXG]])
    // CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK-SAME: num_heads = 8
    // CHECK-SAME: kv_num_heads = 1
    // CHECK-SAME: no_causal = false
    // CHECK-NOT: onnx.Attention

    return %out#0, %out#1, %out#2
        : tensor<?x?x2048xf16>, tensor<?x1x?x256xf16>,
          tensor<?x1x?x256xf16>
  }

  // Single-output (Y only) causal self-attention, no past, no mask. hip.gqa
  // always writes present_key/present_value, so internal DPS present buffers are
  // synthesized ([B, kv_heads, ?, head_dim]) and their results dropped. Built-in
  // causal mask applies (no_causal = false).
  func.func @attn_1out_causal(
      %query: tensor<1x8x2048xf16>,
      %key: tensor<1x8x1024xf16>,
      %value: tensor<1x8x1024xf16>)
      -> tensor<1x8x2048xf16> {

    // CHECK-LABEL: func.func @attn_1out_causal
    // CHECK-SAME: (%[[CTX3:.*]]: !hip.context,

    %y = "onnx.Attention"(%query, %key, %value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 8 : si64,
         is_causal = 1 : si64,
         scale = 0.0883883461 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<1x8x2048xf16>, tensor<1x8x1024xf16>, tensor<1x8x1024xf16>)
        -> tensor<1x8x2048xf16>

    // Synthesized present buffers (seq dynamic = current tokens), head_dim = 128.
    // CHECK: tensor.empty(%{{.*}}) : tensor<1x8x?x128xf16>
    // CHECK: hip.gqa(%[[CTX3]])
    // CHECK-SAME: num_heads = 16
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: no_causal = false
    // CHECK-NOT: onnx.Attention

    return %y : tensor<1x8x2048xf16>
  }

  // Bidirectional (encoder) self-attention: is_causal=0 with NO mask and NO
  // past -> no_causal = true, no attention_bias (runtime bidirectional no-past
  // path). Single output; present synthesized and dropped.
  func.func @attn_bidirectional(
      %query: tensor<2x10x1024xf16>,
      %key: tensor<2x10x1024xf16>,
      %value: tensor<2x10x1024xf16>)
      -> tensor<2x10x1024xf16> {

    // CHECK-LABEL: func.func @attn_bidirectional
    // CHECK-SAME: (%[[CTX4:.*]]: !hip.context,

    %y = "onnx.Attention"(%query, %key, %value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 16 : si64,
         is_causal = 0 : si64,
         scale = 1.250000e-01 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<2x10x1024xf16>, tensor<2x10x1024xf16>, tensor<2x10x1024xf16>)
        -> tensor<2x10x1024xf16>

    // CHECK: hip.gqa(%[[CTX4]])
    // CHECK-SAME: num_heads = 16
    // CHECK-SAME: kv_num_heads = 16
    // CHECK-SAME: no_causal = true
    // CHECK-NOT: onnx.Attention

    return %y : tensor<2x10x1024xf16>
  }

  // Rank-4 BNSH Q/K/V (head counts inferred from shape, attrs omitted), external
  // fp16 mask, no past, single output. Q/K/V are transposed+collapsed to rank-3
  // BSHD before hip.gqa; the rank-3 output is expanded+transposed back to BNSH.
  func.func @attn_rank4(
      %query: tensor<1x16x8x64xf16>,
      %key: tensor<1x8x8x64xf16>,
      %value: tensor<1x8x8x64xf16>,
      %attn_mask: tensor<1x1x8x8xf16>)
      -> tensor<1x16x8x64xf16> {

    // CHECK-LABEL: func.func @attn_rank4
    // CHECK-SAME: (%[[CTX5:.*]]: !hip.context,

    %y = "onnx.Attention"(%query, %key, %value, %attn_mask)
        {is_causal = 0 : si64,
         scale = 1.250000e-01 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<1x16x8x64xf16>, tensor<1x8x8x64xf16>, tensor<1x8x8x64xf16>,
           tensor<1x1x8x8xf16>)
        -> tensor<1x16x8x64xf16>

    // Input BNSH -> BSHD: transpose(perm=[0,2,1,3]) then collapse to rank-3.
    // CHECK: hip.transpose(%[[CTX5]]) ins(%{{.*}} : tensor<1x16x8x64xf16>)
    // CHECK-SAME: perm = [0, 2, 1, 3]
    // CHECK: tensor.collapse_shape
    // CHECK: hip.gqa(%[[CTX5]])
    // CHECK-SAME: num_heads = 16
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: no_causal = true
    // Output BSHD -> BNSH: expand to rank-4 then transpose(perm=[0,2,1,3]).
    // CHECK: tensor.expand_shape
    // CHECK: hip.transpose(%[[CTX5]])
    // CHECK-SAME: perm = [0, 2, 1, 3]
    // CHECK-NOT: onnx.Attention

    return %y : tensor<1x16x8x64xf16>
  }
}
