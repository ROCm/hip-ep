// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Lock in the present_key/present_value DPS init sizing when the present seq
// dim is DYNAMIC.
//
// present KV is concat(past, current) along the seq axis, so the present seq
// extent is past_seq + current_seq. Sizing it from the past operand alone
// collapses present to a zero-length buffer on a fresh prefill (past_seq == 0);
// the output allocator then hands the runtime a null present_key/present_value
// and wrap_group_query_attention rejects the call, leaving attention output
// zero-filled (observed as gibberish text on gpt-oss-20b).
//
// The other GQA lit tests all use fully static present shapes, where the init
// is a plain tensor.empty() with no dynamic operands and the sizing source is
// therefore unobservable. This test uses dynamic batch/seq so the computed
// extents appear in the IR.
//
// Companion: OnnxAttentionConversion.cpp applies the same %tot = %past + %cur
// rule for onnx.Attention.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===========================================================================
  // Dynamic batch + dynamic seq, separate K/V, dynamic past KV.
  // Function must be named @main_graph for --hip-add-context-arg to fire.
  // ===========================================================================
  func.func @main_graph(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>,
      %past_key: tensor<?x8x?x128xf16>,
      %past_value: tensor<?x8x?x128xf16>,
      %seqlens_k: tensor<?x1xi32>,
      %total_seq_len: tensor<i32>)
      -> (tensor<?x?x4096xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !hip.context,
    // CHECK-SAME: %[[Q:[a-zA-Z0-9_]+]]: tensor<?x?x4096xf16>,
    // CHECK-SAME: %[[PK:[a-zA-Z0-9_]+]]: tensor<?x8x?x128xf16>,

    %none1 = "onnx.NoValue"() {value} : () -> none
    %none2 = "onnx.NoValue"() {value} : () -> none

    %out:3 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                            %seqlens_k, %total_seq_len, %none1, %none2)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 32 : si64,
         kv_num_heads = 8 : si64,
         scale = 0.0883883461 : f32,
         softcap = 0.000000e+00 : f32,
         do_rotary = 0 : si64,
         rotary_interleaved = 0 : si64}
        : (tensor<?x?x4096xf16>, tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
           tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>,
           tensor<?x1xi32>, tensor<i32>, none, none)
        -> (tensor<?x?x4096xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>)

    // The output init comes first; anchor on it so the dims below are the
    // present-init ones.
    // CHECK: tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x4096xf16>

    // present seq extent must be past + current, NOT past alone.
    // current KV tokens == query seq length (rank-3 BSH, dim 1);
    // valid past length == dim(past_key, 2) (rank-4 BNSH).
    // CHECK: %[[B:.*]] = tensor.dim %[[Q]], %{{.*}} : tensor<?x?x4096xf16>
    // CHECK: %[[CUR:.*]] = tensor.dim %[[Q]], %{{.*}} : tensor<?x?x4096xf16>
    // CHECK: %[[PAST:.*]] = tensor.dim %[[PK]], %{{.*}} : tensor<?x8x?x128xf16>
    // CHECK: %[[TOT:.*]] = arith.addi %[[PAST]], %[[CUR]] : index
    // CHECK: tensor.empty(%[[B]], %[[TOT]]) : tensor<?x8x?x128xf16>
    // CHECK: %[[B2:.*]] = tensor.dim %[[Q]], %{{.*}} : tensor<?x?x4096xf16>
    // CHECK: tensor.empty(%[[B2]], %[[TOT]]) : tensor<?x8x?x128xf16>
    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: num_heads = 32
    // CHECK-NOT: onnx.Custom

    return %out#0, %out#1, %out#2 : tensor<?x?x4096xf16>, tensor<?x8x?x128xf16>, tensor<?x8x?x128xf16>
  }
}
