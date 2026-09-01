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
//   - a sliding window baked into the additive mask recovered from the mask's
//     producing subgraph and stamped as local_window_size, since
//     onnx.Attention has no attribute to carry one
//
// Cases: a static-shape decode (16 query heads, 8 KV heads), a dynamic-shape
// prefill with an EMPTY past (the case that previously sized present from
// dim(past_key, 2) alone -> zero-length present buffer -> null present_key ->
// zeroed attention output), the Gemma-4-E2B decoder self-attn (is_causal=1 +
// fp16 mask + past KV + 3 outputs -- previously rejected), a single-output
// causal case, a bidirectional no-mask case, a rank-4 BNSH case, and the
// window-recovery cases: windowed layer, global layer, unrecognized OR leg,
// the segment leg stripped of its CumSum provenance, the window leg's indices
// stripped of their shared position source, and the window leg's indices drawn
// from two DIFFERENT position sources.
// ============================================================================

// The pass runs `--convert-onnx-to-hip`, whose module-metadata step requires
// a `@main_graph` entry function per module; each scenario is therefore its
// own `--split-input-file` section named `@main_graph`.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --split-input-file | FileCheck %s

// ===== Static-shape decode: 16 query heads, 8 KV heads, external mask =====
module {
  func.func @main_graph(
      %query: tensor<1x1x2048xf16>,
      %key: tensor<1x1x1024xf16>,
      %value: tensor<1x1x1024xf16>,
      %attn_mask: tensor<1x1x1x128xf16>,
      %past_key: tensor<1x8x127x128xf16>,
      %past_value: tensor<1x8x127x128xf16>)
      -> (tensor<1x1x2048xf16>, tensor<1x8x128x128xf16>,
          tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @main_graph
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
    // Mask is threaded as the final hip.gqa input (attention_bias).
    // CHECK-SAME: ins({{.*}}tensor<1x1x1x128xf16>) outs
    // Attributes print alphabetically; no_causal = true is present (is_causal=0).
    // CHECK-SAME: {kv_num_heads = 8 : i64, no_causal = true, num_heads = 16 : i64
    // CHECK-NOT: onnx.Attention

    return %out#0, %out#1, %out#2
        : tensor<1x1x2048xf16>, tensor<1x8x128x128xf16>,
          tensor<1x8x128x128xf16>
  }
}

// -----

// Dynamic-shape prefill with an EMPTY past: present seq is computed at
// runtime as past_seq + current_seq (NOT dim(past_key, 2) alone), so the
// present buffer is sized to the real KV length even when past_seq == 0.
// Mirrors the Gemma-4 decoder self-attn (16 query heads, 8 KV heads, d=256).
module {
  func.func @main_graph(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x2048xf16>,
      %value: tensor<?x?x2048xf16>,
      %attn_mask: tensor<?x1x?x?xf16>,
      %past_key: tensor<?x8x?x256xf16>,
      %past_value: tensor<?x8x?x256xf16>)
      -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
          tensor<?x8x?x256xf16>) {

    // CHECK-LABEL: func.func @main_graph
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

// -----

// Gemma-4-E2B decoder self-attn: is_causal=1 WITH an external fp16 additive
// mask, past KV cache, 3 outputs, rank-3 Q/K/V (q_num_heads=8, kv_num_heads=1,
// head_dim=256). This exact combination (causal + explicit mask) was
// previously REJECTED by the converter, leaving onnx.Attention unconverted ->
// "op was not bufferized". Now it converts: no_causal=false (built-in causal
// still applies) AND the mask is threaded as attention_bias (added first).
module {
  func.func @main_graph(
      %query: tensor<?x?x2048xf16>,
      %key: tensor<?x?x256xf16>,
      %value: tensor<?x?x256xf16>,
      %attn_mask: tensor<?x1x?x?xf16>,
      %past_key: tensor<?x1x?x256xf16>,
      %past_value: tensor<?x1x?x256xf16>)
      -> (tensor<?x?x2048xf16>, tensor<?x1x?x256xf16>,
          tensor<?x1x?x256xf16>) {

    // CHECK-LABEL: func.func @main_graph
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
    // Mask is threaded as the final hip.gqa input (attention_bias).
    // CHECK-SAME: ins({{.*}}tensor<?x1x?x?xf16>) outs
    // Attributes print alphabetically; no_causal defaults to false and is
    // omitted, so the dict goes straight from kv_num_heads to num_heads.
    // CHECK-SAME: {kv_num_heads = 1 : i64, num_heads = 8 : i64
    // CHECK-NOT: onnx.Attention

    return %out#0, %out#1, %out#2
        : tensor<?x?x2048xf16>, tensor<?x1x?x256xf16>,
          tensor<?x1x?x256xf16>
  }
}

// -----

// Single-output (Y only) causal self-attention, no past, no mask. hip.gqa
// always writes present_key/present_value, so internal DPS present buffers are
// synthesized ([B, kv_heads, ?, head_dim]) and their results dropped. Built-in
// causal mask applies (no_causal = false).
module {
  func.func @main_graph(
      %query: tensor<1x8x2048xf16>,
      %key: tensor<1x8x1024xf16>,
      %value: tensor<1x8x1024xf16>)
      -> tensor<1x8x2048xf16> {

    // CHECK-LABEL: func.func @main_graph
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
    // Attributes print alphabetically; no_causal defaults to false and is
    // omitted (is_causal=1), so the dict goes straight from kv_num_heads to
    // num_heads.
    // CHECK-SAME: {kv_num_heads = 8 : i64, num_heads = 16 : i64
    // CHECK-NOT: onnx.Attention

    return %y : tensor<1x8x2048xf16>
  }
}

// -----

// Bidirectional (encoder) self-attention: is_causal=0 with NO mask and NO
// past -> no_causal = true, no attention_bias (runtime bidirectional no-past
// path). Single output; present synthesized and dropped.
module {
  func.func @main_graph(
      %query: tensor<2x10x1024xf16>,
      %key: tensor<2x10x1024xf16>,
      %value: tensor<2x10x1024xf16>)
      -> tensor<2x10x1024xf16> {

    // CHECK-LABEL: func.func @main_graph
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
    // Attributes print alphabetically; no_causal = true is present (is_causal=0).
    // CHECK-SAME: {kv_num_heads = 16 : i64, no_causal = true, num_heads = 16 : i64
    // CHECK-NOT: onnx.Attention

    return %y : tensor<2x10x1024xf16>
  }
}

// -----

// Rank-4 BNSH Q/K/V (head counts inferred from shape, attrs omitted), external
// fp16 mask, no past, single output. Q/K/V are transposed+collapsed to rank-3
// BSHD before hip.gqa; the rank-3 output is expanded+transposed back to BNSH.
module {
  func.func @main_graph(
      %query: tensor<1x16x8x64xf16>,
      %key: tensor<1x8x8x64xf16>,
      %value: tensor<1x8x8x64xf16>,
      %attn_mask: tensor<1x1x8x8xf16>)
      -> tensor<1x16x8x64xf16> {

    // CHECK-LABEL: func.func @main_graph
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
    // Attributes print alphabetically; no_causal = true is present (is_causal=0).
    // CHECK-SAME: {kv_num_heads = 8 : i64, no_causal = true, num_heads = 16 : i64
    // Output BSHD -> BNSH: expand to rank-4 then transpose(perm=[0,2,1,3]).
    // CHECK: tensor.expand_shape
    // CHECK: hip.transpose(%[[CTX5]])
    // CHECK-SAME: perm = [0, 2, 1, 3]
    // CHECK-NOT: onnx.Attention

    return %y : tensor<1x16x8x64xf16>
  }
}

// -----

// ===== Sliding window recovered from the mask subgraph (windowed layer) =====
//
// Before opset 25 there was no window attribute on `onnx.Attention`, so a
// windowed model could only express its window in the additive mask.
// AttentionWindowFold reads it back out and stamps it, which is what lets the
// runtime narrow its key range instead of scoring everything.
//
// This is Gemma-4's local-layer mask, reproduced node for node: the keep
// condition ANDs the padding mask with an OR of the windowed-causal leg and the
// same-image-block bidirectional leg. The window leg is
// `And(qpos >= kpos, qpos - kpos < 1024)`, and the compare and the subtraction
// reference the SAME two values in the same order -- that identity is what
// makes `qpos` the query side and `kpos` the key side.
//
// Both index tensors are then built from ONE onnx.CumSum, which is the other
// half of what the matcher requires: identity alone bounds the difference of
// two tensors, and only a shared position source makes that difference a
// position distance in units of key positions. Two later cases strip this,
// once entirely and once by giving each side its own source.
//
// The segment leg's block index is built through a SEPARATE onnx.CumSum, as it
// is in the real export -- which has exactly two, a block-index one behind the
// equality and a position one behind the indices. The matcher requires that
// ancestor too, so removing it alone is enough to make the whole match decline.
module {
  func.func @main_graph(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x2048xf16>,
      %value: tensor<?x?x2048xf16>,
      %attnmask: tensor<?x?xi64>,
      %imgstart: tensor<?x?xi64>,
      %pad: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x256xf16>,
      %past_value: tensor<?x8x?x256xf16>)
      -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
          tensor<?x8x?x256xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTXW:.*]]: !hip.context,

    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %w = "onnx.Constant"() {value = dense<1024> : tensor<i64>}
        : () -> tensor<i64>
    %zi = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>}
        : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>}
        : () -> tensor<f32>

    // Positions, built as the real export builds them: a cumulative count over
    // the attention mask numbers the keys, and the query positions are the
    // trailing entries of that SAME sequence. On the export the key side
    // reaches this CumSum in one hop and the query side in two, through the
    // Slice; kPositionProvenanceDepth is 6, so both are well inside it.
    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kidx = "onnx.CumSum"(%attnmask, %ax1)
        : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %sbeg = "onnx.Constant"() {value = dense<-8> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %send = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %qidx = "onnx.Slice"(%kidx, %sbeg, %send, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<?x?xi64>

    // Broadcast the two index vectors into [B, q, k]: the query index goes on
    // the lower axis (unsqueezed at 2), the key index on the higher (at 1).
    %qpos = "onnx.Unsqueeze"(%qidx, %c2)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Unsqueeze"(%kidx, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>

    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %dist = "onnx.Sub"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi64>
    %inwin = "onnx.Less"(%dist, %w)
        : (tensor<?x?x?xi64>, tensor<i64>) -> tensor<?x?x?xi1>
    %wleg = "onnx.And"(%causal, %inwin)
        : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>

    // Same-image-block bidirectional leg: identically false for a text-only
    // prompt, and bounded by the image block length otherwise.
    //
    // The block index is a running count of image-block starts, so an
    // onnx.CumSum sits behind the equality -- a different one from the position
    // CumSum above. That is the provenance the matcher requires: it is what
    // distinguishes a block index from a document id or a prefix-LM boundary,
    // which wear the same Equal shape but bound nothing.
    %blk = "onnx.CumSum"(%imgstart, %ax1)
        : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %qblk = "onnx.Unsqueeze"(%blk, %c2)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kblk = "onnx.Unsqueeze"(%blk, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %sameblk = "onnx.Equal"(%qblk, %kblk)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %isimg = "onnx.GreaterOrEqual"(%qblk, %zi)
        : (tensor<?x?x1xi64>, tensor<i64>) -> tensor<?x?x1xi1>
    %segleg = "onnx.And"(%sameblk, %isimg)
        : (tensor<?x?x?xi1>, tensor<?x?x1xi1>) -> tensor<?x?x?xi1>

    %or = "onnx.Or"(%wleg, %segleg)
        : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %padu = "onnx.Unsqueeze"(%pad, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64}
        : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %or)
        : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64}
        : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1)
        : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>

    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key,
                              %past_value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 8 : si64,
         is_causal = 0 : si64,
         scale = 6.250000e-02 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<?x?x4096xf16>, tensor<?x?x2048xf16>, tensor<?x?x2048xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x256xf16>, tensor<?x8x?x256xf16>)
        -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
            tensor<?x8x?x256xf16>)

    // CHECK: hip.gqa(%[[CTXW]])
    // Attributes print alphabetically, so the recovered window lands between
    // kv_num_heads and no_causal.
    // CHECK-SAME: {kv_num_heads = 8 : i64, local_window_size = 1024 : i64, no_causal = true, num_heads = 16 : i64
    // CHECK-NOT: onnx.Attention

    return %out#0, %out#1, %out#2
        : tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>, tensor<?x8x?x256xf16>
  }
}

// -----

// ===== Global layer: the same mask WITHOUT the window leg =====
//
// Gemma-4's 5 full-attention layers differ from its 25 windowed ones by exactly
// one node: the OR's causal leg is the bare compare rather than
// `And(causal, within-window)`. A bare causal term bounds nothing from below, so
// no window is recovered and the op keeps hip.gqa's -1 default. This is the
// case that must NOT be stamped -- forcing a window here would change results
// rather than just make them cheaper.
module {
  func.func @main_graph(
      %query: tensor<?x?x8192xf16>,
      %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>,
      %qidx: tensor<?x?xi64>,
      %kidx: tensor<?x?xi64>,
      %pad: tensor<?x?xi64>,
      %past_key: tensor<?x2x?x512xf16>,
      %past_value: tensor<?x2x?x512xf16>)
      -> (tensor<?x?x8192xf16>, tensor<?x2x?x512xf16>,
          tensor<?x2x?x512xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTXG2:.*]]: !hip.context,

    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>}
        : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>}
        : () -> tensor<f32>

    %qpos = "onnx.Unsqueeze"(%qidx, %c2)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Unsqueeze"(%kidx, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>

    %padu = "onnx.Unsqueeze"(%pad, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64}
        : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %causal)
        : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64}
        : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1)
        : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>

    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key,
                              %past_value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 2 : si64,
         is_causal = 0 : si64,
         scale = 4.419420e-02 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<?x?x8192xf16>, tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
           tensor<?x1x?x?xf16>, tensor<?x2x?x512xf16>, tensor<?x2x?x512xf16>)
        -> (tensor<?x?x8192xf16>, tensor<?x2x?x512xf16>,
            tensor<?x2x?x512xf16>)

    // No local_window_size: the dict goes straight from kv_num_heads to
    // no_causal, so the attribute is absent and the -1 default stands.
    // CHECK: hip.gqa(%[[CTXG2]])
    // CHECK-SAME: {kv_num_heads = 2 : i64, no_causal = true, num_heads = 16 : i64
    // CHECK-NOT: onnx.Attention

    return %out#0, %out#1, %out#2
        : tensor<?x?x8192xf16>, tensor<?x2x?x512xf16>, tensor<?x2x?x512xf16>
  }
}

// -----

// ===== An unrecognized OR leg must abandon the window =====
//
// A disjunct ADDS keeps, so one that cannot be bounded leaves the whole
// condition unbounded no matter how tight the window leg is: the model may be
// keeping a key the narrowing would drop. Here the second leg is a bare
// equality on the position indices, which is neither a window nor the
// recognized same-segment shape, so nothing is stamped even though the window
// leg itself is a perfect match -- indices share a position source and all.
// Recognizing only what it understands is the whole safety property of this
// rewrite.
module {
  func.func @main_graph(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x2048xf16>,
      %value: tensor<?x?x2048xf16>,
      %attnmask: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x256xf16>,
      %past_value: tensor<?x8x?x256xf16>)
      -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
          tensor<?x8x?x256xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTXU:.*]]: !hip.context,

    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %w = "onnx.Constant"() {value = dense<1024> : tensor<i64>}
        : () -> tensor<i64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>}
        : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>}
        : () -> tensor<f32>

    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kidx = "onnx.CumSum"(%attnmask, %ax1)
        : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %sbeg = "onnx.Constant"() {value = dense<-8> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %send = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %qidx = "onnx.Slice"(%kidx, %sbeg, %send, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<?x?xi64>

    %qpos = "onnx.Unsqueeze"(%qidx, %c2)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Unsqueeze"(%kidx, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %dist = "onnx.Sub"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi64>
    %inwin = "onnx.Less"(%dist, %w)
        : (tensor<?x?x?xi64>, tensor<i64>) -> tensor<?x?x?xi1>
    %wleg = "onnx.And"(%causal, %inwin)
        : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>

    %other = "onnx.Equal"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %keep = "onnx.Or"(%wleg, %other)
        : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64}
        : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1)
        : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>

    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key,
                              %past_value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 8 : si64,
         is_causal = 0 : si64,
         scale = 6.250000e-02 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<?x?x4096xf16>, tensor<?x?x2048xf16>, tensor<?x?x2048xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x256xf16>, tensor<?x8x?x256xf16>)
        -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
            tensor<?x8x?x256xf16>)

    // CHECK: hip.gqa(%[[CTXU]])
    // CHECK-SAME: {kv_num_heads = 8 : i64, no_causal = true, num_heads = 16 : i64
    // CHECK-NOT: onnx.Attention

    return %out#0, %out#1, %out#2
        : tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>, tensor<?x8x?x256xf16>
  }
}

// -----

// ===== The segment shape WITHOUT its CumSum provenance is declined =====
//
// Node for node the windowed case above, except the block index comes straight
// out of a graph input instead of through an onnx.CumSum. `And(Equal(a,b),
// a >= 0)` is also how a same-document mask and a prefix-LM mask are spelled,
// and those reach arbitrarily far back, so the shape alone is not enough to
// accept: without a running count behind the equality the leg is unrecognized,
// the OR it sits under becomes unbounded, and no window is stamped even though
// the window leg is a perfect match, position source included. This is the
// negative half of the segment provenance check -- the positive half is the
// windowed case above, which is identical but for that CumSum.
module {
  func.func @main_graph(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x2048xf16>,
      %value: tensor<?x?x2048xf16>,
      %attnmask: tensor<?x?xi64>,
      %docid: tensor<?x?xi64>,
      %pad: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x256xf16>,
      %past_value: tensor<?x8x?x256xf16>)
      -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
          tensor<?x8x?x256xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTXNP:.*]]: !hip.context,

    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %w = "onnx.Constant"() {value = dense<1024> : tensor<i64>}
        : () -> tensor<i64>
    %zi = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>}
        : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>}
        : () -> tensor<f32>

    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %kidx = "onnx.CumSum"(%attnmask, %ax1)
        : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %sbeg = "onnx.Constant"() {value = dense<-8> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %send = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %qidx = "onnx.Slice"(%kidx, %sbeg, %send, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<?x?xi64>

    %qpos = "onnx.Unsqueeze"(%qidx, %c2)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Unsqueeze"(%kidx, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %dist = "onnx.Sub"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi64>
    %inwin = "onnx.Less"(%dist, %w)
        : (tensor<?x?x?xi64>, tensor<i64>) -> tensor<?x?x?xi1>
    %wleg = "onnx.And"(%causal, %inwin)
        : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>

    // Same shape as the recognized segment leg, but %docid is an input rather
    // than a running count.
    %qdoc = "onnx.Unsqueeze"(%docid, %c2)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kdoc = "onnx.Unsqueeze"(%docid, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %samedoc = "onnx.Equal"(%qdoc, %kdoc)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %isdoc = "onnx.GreaterOrEqual"(%qdoc, %zi)
        : (tensor<?x?x1xi64>, tensor<i64>) -> tensor<?x?x1xi1>
    %docleg = "onnx.And"(%samedoc, %isdoc)
        : (tensor<?x?x?xi1>, tensor<?x?x1xi1>) -> tensor<?x?x?xi1>

    %or = "onnx.Or"(%wleg, %docleg)
        : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %padu = "onnx.Unsqueeze"(%pad, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64}
        : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %or)
        : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64}
        : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1)
        : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>

    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key,
                              %past_value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 8 : si64,
         is_causal = 0 : si64,
         scale = 6.250000e-02 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<?x?x4096xf16>, tensor<?x?x2048xf16>, tensor<?x?x2048xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x256xf16>, tensor<?x8x?x256xf16>)
        -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
            tensor<?x8x?x256xf16>)

    // No local_window_size: the dict goes straight from kv_num_heads to
    // no_causal.
    // CHECK: hip.gqa(%[[CTXNP]])
    // CHECK-SAME: {kv_num_heads = 8 : i64, no_causal = true, num_heads = 16 : i64
    // CHECK-NOT: onnx.Attention

    return %out#0, %out#1, %out#2
        : tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>, tensor<?x8x?x256xf16>
  }
}

// -----

// ===== A window leg with no position source is declined =====
//
// `And(p >= q, p - q < 1024)` on two tensors that arrive straight out of graph
// inputs. The shape is a flawless window leg and the same-value identity holds
// on both halves, so this is what the matcher accepted before it asked where
// the indices came from.
//
// It is not enough. The identity proves `p - q` is bounded by 1024; the runtime
// turns that into `kv_lo = abs_q - 1024 + 1`, an ABSOLUTE KEY POSITION. That
// step needs p and q to be positions on one common scale, and nothing here says
// they are -- indices scaled by a block size, or re-based per segment, wear
// exactly this shape and would hand over a W in the wrong units, dropping keys
// the model wanted. So the leg is unrecognized, the And above it has no bounded
// conjunct, and no window is stamped.
//
// There is no OR leg and no segment leg in this mask, so the decline is
// attributable to the missing provenance and nothing else.
module {
  func.func @main_graph(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x2048xf16>,
      %value: tensor<?x?x2048xf16>,
      %qidx: tensor<?x?xi64>,
      %kidx: tensor<?x?xi64>,
      %pad: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x256xf16>,
      %past_value: tensor<?x8x?x256xf16>)
      -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
          tensor<?x8x?x256xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTXNS:.*]]: !hip.context,

    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %w = "onnx.Constant"() {value = dense<1024> : tensor<i64>}
        : () -> tensor<i64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>}
        : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>}
        : () -> tensor<f32>

    %qpos = "onnx.Unsqueeze"(%qidx, %c2)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Unsqueeze"(%kidx, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %dist = "onnx.Sub"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi64>
    %inwin = "onnx.Less"(%dist, %w)
        : (tensor<?x?x?xi64>, tensor<i64>) -> tensor<?x?x?xi1>
    %wleg = "onnx.And"(%causal, %inwin)
        : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>

    %padu = "onnx.Unsqueeze"(%pad, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64}
        : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %wleg)
        : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64}
        : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1)
        : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>

    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key,
                              %past_value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 8 : si64,
         is_causal = 0 : si64,
         scale = 6.250000e-02 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<?x?x4096xf16>, tensor<?x?x2048xf16>, tensor<?x?x2048xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x256xf16>, tensor<?x8x?x256xf16>)
        -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
            tensor<?x8x?x256xf16>)

    // CHECK: hip.gqa(%[[CTXNS]])
    // CHECK-SAME: {kv_num_heads = 8 : i64, no_causal = true, num_heads = 16 : i64
    // CHECK-NOT: onnx.Attention

    return %out#0, %out#1, %out#2
        : tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>, tensor<?x8x?x256xf16>
  }
}

// -----

// ===== Two DIFFERENT position sources are declined =====
//
// The sharp case, and the reason the check intersects sources rather than just
// requiring each side to have one. Both indices are cumulative counts here, so
// "is there a CumSum above this value" is true of each in isolation -- but they
// are counts over two different tensors, so they are two unrelated numberings.
//
// Subtracting across them measures nothing: `p - q < 1024` says the two counts
// are within 1024 of each other, which does not bound how far back in the KEY
// sequence a query reaches. Narrowing on it would be unsound, so the leg is
// unrecognized and no window is stamped.
module {
  func.func @main_graph(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x2048xf16>,
      %value: tensor<?x?x2048xf16>,
      %maska: tensor<?x?xi64>,
      %maskb: tensor<?x?xi64>,
      %pad: tensor<?x?xi64>,
      %past_key: tensor<?x8x?x256xf16>,
      %past_value: tensor<?x8x?x256xf16>)
      -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
          tensor<?x8x?x256xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTXTS:.*]]: !hip.context,

    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %c2 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %w = "onnx.Constant"() {value = dense<1024> : tensor<i64>}
        : () -> tensor<i64>
    %keepv = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>}
        : () -> tensor<f32>
    %dropv = "onnx.Constant"() {value = dense<-6.550400e+04> : tensor<f32>}
        : () -> tensor<f32>

    // Two CumSums, two numberings. Each side would pass a per-side provenance
    // test; the pair does not share a source, which is what actually matters.
    %ax1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %qidx = "onnx.CumSum"(%maska, %ax1)
        : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>
    %kidx = "onnx.CumSum"(%maskb, %ax1)
        : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi64>

    %qpos = "onnx.Unsqueeze"(%qidx, %c2)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x?x1xi64>
    %kpos = "onnx.Unsqueeze"(%kidx, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %causal = "onnx.GreaterOrEqual"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi1>
    %dist = "onnx.Sub"(%qpos, %kpos)
        : (tensor<?x?x1xi64>, tensor<?x1x?xi64>) -> tensor<?x?x?xi64>
    %inwin = "onnx.Less"(%dist, %w)
        : (tensor<?x?x?xi64>, tensor<i64>) -> tensor<?x?x?xi1>
    %wleg = "onnx.And"(%causal, %inwin)
        : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>

    %padu = "onnx.Unsqueeze"(%pad, %c1)
        : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
    %padb = "onnx.Cast"(%padu) {to = 9 : si64}
        : (tensor<?x1x?xi64>) -> tensor<?x1x?xi1>
    %keep = "onnx.And"(%padb, %wleg)
        : (tensor<?x1x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %bias32 = "onnx.Where"(%keep, %keepv, %dropv)
        : (tensor<?x?x?xi1>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    %bias16 = "onnx.Cast"(%bias32) {to = 10 : si64}
        : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16>
    %bias = "onnx.Unsqueeze"(%bias16, %c1)
        : (tensor<?x?x?xf16>, tensor<1xi64>) -> tensor<?x1x?x?xf16>

    %out:3 = "onnx.Attention"(%query, %key, %value, %bias, %past_key,
                              %past_value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 8 : si64,
         is_causal = 0 : si64,
         scale = 6.250000e-02 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<?x?x4096xf16>, tensor<?x?x2048xf16>, tensor<?x?x2048xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x256xf16>, tensor<?x8x?x256xf16>)
        -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
            tensor<?x8x?x256xf16>)

    // CHECK: hip.gqa(%[[CTXTS]])
    // CHECK-SAME: {kv_num_heads = 8 : i64, no_causal = true, num_heads = 16 : i64
    // CHECK-NOT: onnx.Attention

    return %out#0, %out#1, %out#2
        : tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>, tensor<?x8x?x256xf16>
  }
}
