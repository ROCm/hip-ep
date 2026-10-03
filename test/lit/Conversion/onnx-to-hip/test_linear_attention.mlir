// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify com.microsoft.LinearAttention (via onnx.Custom) is correctly
// lowered to hip.linear_attention operation in tensor-first mode.
//
// This test suite validates:
// - Custom ONNX op matching by function_name and domain_name
// - Multi-output op lowering (2 outputs: output, present_state)
// - Attribute preservation (q_num_heads, kv_num_heads, scale, update_rule, etc.)
// - Optional inputs handling (past_state, decay, beta)
// - f16 element type support
// - Tensor-first DPS: tensor.empty() for each output
//
// Test cases:
// 1. Minimal linear mode (3 required inputs only)
// 2. Gated mode with decay
// 3. Delta mode with beta
// 4. Full gated_delta mode (all 6 inputs + all attributes)
// 5. Dynamic batch + seq_len (prefill case, no past_state)
// 6. Dynamic batch + seq_len with past_state (typical LLM decode with KV-cache)
//
// Dynamic cases additionally assert:
// - tensor.dim extracts each dynamic dim at runtime
// - tensor.empty(%dim...) takes those sizes as dynamic operands
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

// =============================================================================
// Test 1: Minimal linear mode - 3 required inputs only
// =============================================================================
module {
  func.func @main_graph(
      %query: tensor<1x128x4096xf16>,
      %key: tensor<1x128x1024xf16>,
      %value: tensor<1x128x1024xf16>)
      -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %out:2 = "onnx.Custom"(%query, %key, %value)
        <{function_name = "LinearAttention"}>
        {domain_name = "com.microsoft",
         q_num_heads = 32 : si64,
         kv_num_heads = 8 : si64,
         update_rule = "linear"}
        : (tensor<1x128x4096xf16>, tensor<1x128x1024xf16>,
           tensor<1x128x1024xf16>)
        -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>)

    // CHECK: tensor.empty() : tensor<1x128x4096xf16>
    // CHECK: tensor.empty() : tensor<1x8x128x128xf16>
    // CHECK: hip.linear_attention(%[[CTX]])
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: q_num_heads = 32
    // CHECK-SAME: update_rule = "linear"

    return %out#0, %out#1 : tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>
  }

  // ===========================================================================
  // Test 2: Gated mode with past_state and decay
  // ===========================================================================
  func.func @test_gated_with_decay(
      %query: tensor<1x1x4096xf16>,
      %key: tensor<1x1x1024xf16>,
      %value: tensor<1x1x1024xf16>,
      %past_state: tensor<1x8x128x128xf16>,
      %decay: tensor<1x1x1024xf16>)
      -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @test_gated_with_decay
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %none = "onnx.NoValue"() {value} : () -> none

    %out:2 = "onnx.Custom"(%query, %key, %value, %past_state, %decay, %none)
        <{function_name = "LinearAttention"}>
        {domain_name = "com.microsoft",
         q_num_heads = 32 : si64,
         kv_num_heads = 8 : si64,
         update_rule = "gated"}
        : (tensor<1x1x4096xf16>, tensor<1x1x1024xf16>,
           tensor<1x1x1024xf16>, tensor<1x8x128x128xf16>,
           tensor<1x1x1024xf16>, none)
        -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>)

    // CHECK: hip.linear_attention(%[[CTX]])
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: q_num_heads = 32
    // CHECK-SAME: update_rule = "gated"

    return %out#0, %out#1 : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>
  }

  // ===========================================================================
  // Test 3: Delta mode with past_state and beta
  // ===========================================================================
  func.func @test_delta_with_beta(
      %query: tensor<1x1x4096xf16>,
      %key: tensor<1x1x1024xf16>,
      %value: tensor<1x1x1024xf16>,
      %past_state: tensor<1x8x128x128xf16>,
      %beta: tensor<1x1x8xf16>)
      -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @test_delta_with_beta
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %none = "onnx.NoValue"() {value} : () -> none

    %out:2 = "onnx.Custom"(%query, %key, %value, %past_state, %none, %beta)
        <{function_name = "LinearAttention"}>
        {domain_name = "com.microsoft",
         q_num_heads = 32 : si64,
         kv_num_heads = 8 : si64,
         update_rule = "delta"}
        : (tensor<1x1x4096xf16>, tensor<1x1x1024xf16>,
           tensor<1x1x1024xf16>, tensor<1x8x128x128xf16>,
           none, tensor<1x1x8xf16>)
        -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>)

    // CHECK: hip.linear_attention(%[[CTX]])
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: q_num_heads = 32
    // CHECK-SAME: update_rule = "delta"

    return %out#0, %out#1 : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>
  }

  // ===========================================================================
  // Test 4: Full gated_delta mode - all 6 inputs + all attributes
  // ===========================================================================
  func.func @test_gated_delta_full(
      %query: tensor<1x128x4096xf16>,
      %key: tensor<1x128x1024xf16>,
      %value: tensor<1x128x1024xf16>,
      %past_state: tensor<1x8x128x128xf16>,
      %decay: tensor<1x128x1024xf16>,
      %beta: tensor<1x128x8xf16>)
      -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @test_gated_delta_full
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %out:2 = "onnx.Custom"(%query, %key, %value, %past_state, %decay, %beta)
        <{function_name = "LinearAttention"}>
        {domain_name = "com.microsoft",
         q_num_heads = 32 : si64,
         kv_num_heads = 8 : si64,
         scale = 8.838834e-02 : f32,
         chunk_size = 64 : si64,
         update_rule = "gated_delta"}
        : (tensor<1x128x4096xf16>, tensor<1x128x1024xf16>,
           tensor<1x128x1024xf16>, tensor<1x8x128x128xf16>,
           tensor<1x128x1024xf16>, tensor<1x128x8xf16>)
        -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>)

    // CHECK: tensor.empty() : tensor<1x128x4096xf16>
    // CHECK: tensor.empty() : tensor<1x8x128x128xf16>
    // CHECK: hip.linear_attention(%[[CTX]])
    // CHECK-SAME: ins(
    // CHECK-SAME: chunk_size = 64
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: q_num_heads = 32
    // CHECK-SAME: scale = {{0.088388[0-9]*|8.838834e-02}}
    // "gated_delta" is the default value, so MLIR omits it from output

    return %out#0, %out#1 : tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>
  }

  // ===========================================================================
  // Test 5: Dynamic batch + seq_len, linear mode (prefill without past_state).
  //
  // Typical LLM prefill scenario: batch and seq_len are runtime values; only
  // the hidden dims (H_q*d_k, H_kv*d_k, H_kv*d_v, d_k, d_v) are fixed by the
  // model architecture.
  //
  // Expectations:
  // - output init: tensor.dim for dims 0 and 1 of query + tensor.empty with
  //   two dynamic sizes (output's last dim H_q*d_v stays static).
  // - present_state init: batch is semantically query dim 0 -- the same SSA
  //   extent is reused by both destinations. Remaining dims
  //   (H_kv=8, d_k=128, d_v=128) are static per spec.
  // ===========================================================================
  func.func @test_linear_dynamic_prefill(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>)
      -> (tensor<?x?x4096xf16>, tensor<?x8x128x128xf16>) {

    // CHECK-LABEL: func.func @test_linear_dynamic_prefill
    // CHECK-SAME: (%[[CTX5:.*]]: !hip.context, %[[Q5:.*]]: tensor<?x?x4096xf16>, %[[K5:.*]]: tensor<?x?x1024xf16>, %[[V5:.*]]: tensor<?x?x1024xf16>)

    %out:2 = "onnx.Custom"(%query, %key, %value)
        <{function_name = "LinearAttention"}>
        {domain_name = "com.microsoft",
         q_num_heads = 32 : si64,
         kv_num_heads = 8 : si64,
         update_rule = "linear"}
        : (tensor<?x?x4096xf16>, tensor<?x?x1024xf16>,
           tensor<?x?x1024xf16>)
        -> (tensor<?x?x4096xf16>, tensor<?x8x128x128xf16>)

    // output and state share query's batch extent.
    // CHECK: %[[B5:.*]] = tensor.dim %[[Q5]]
    // CHECK: %[[S5:.*]] = tensor.dim %[[Q5]]
    // CHECK: tensor.empty(%[[B5]], %[[S5]]) : tensor<?x?x4096xf16>
    // CHECK: tensor.empty(%[[B5]]) : tensor<?x8x128x128xf16>
    // CHECK: hip.linear_attention(%[[CTX5]])
    // CHECK-SAME: ins(%[[Q5]], %[[K5]], %[[V5]]
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: q_num_heads = 32
    // CHECK-SAME: update_rule = "linear"

    return %out#0, %out#1 : tensor<?x?x4096xf16>, tensor<?x8x128x128xf16>
  }

  // ===========================================================================
  // Test 6: Dynamic batch + seq_len with past_state (gated_delta decode).
  //
  // past_state must agree with the semantic state shape, but the destination
  // is still built from query/value plus the head-count attributes.
  //
  // Expectations:
  // - output init: tensor.dim for dims 0 and 1 of query + tensor.empty(...).
  // - present_state init reuses query dim 0 + tensor.empty.
  // ===========================================================================
  func.func @test_linear_dynamic_decode_with_state(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x1024xf16>,
      %value: tensor<?x?x1024xf16>,
      %past_state: tensor<?x8x128x128xf16>,
      %decay: tensor<?x?x1024xf16>,
      %beta: tensor<?x?x8xf16>)
      -> (tensor<?x?x4096xf16>, tensor<?x8x128x128xf16>) {

    // CHECK-LABEL: func.func @test_linear_dynamic_decode_with_state
    // CHECK-SAME: (%[[CTX6:.*]]: !hip.context, %[[Q6:.*]]: tensor<?x?x4096xf16>, %[[K6:.*]]: tensor<?x?x1024xf16>, %[[V6:.*]]: tensor<?x?x1024xf16>, %[[PS6:.*]]: tensor<?x8x128x128xf16>

    %out:2 = "onnx.Custom"(%query, %key, %value, %past_state, %decay, %beta)
        <{function_name = "LinearAttention"}>
        {domain_name = "com.microsoft",
         q_num_heads = 32 : si64,
         kv_num_heads = 8 : si64,
         update_rule = "gated_delta"}
        : (tensor<?x?x4096xf16>, tensor<?x?x1024xf16>,
           tensor<?x?x1024xf16>, tensor<?x8x128x128xf16>,
           tensor<?x?x1024xf16>, tensor<?x?x8xf16>)
        -> (tensor<?x?x4096xf16>, tensor<?x8x128x128xf16>)

    // output and state share query's batch extent.
    // CHECK: %[[B6:.*]] = tensor.dim %[[Q6]]
    // CHECK: %[[S6:.*]] = tensor.dim %[[Q6]]
    // CHECK: tensor.empty(%[[B6]], %[[S6]]) : tensor<?x?x4096xf16>
    // CHECK: tensor.empty(%[[B6]]) : tensor<?x8x128x128xf16>
    // CHECK: hip.linear_attention(%[[CTX6]])
    // CHECK-SAME: ins(%[[Q6]], %[[K6]], %[[V6]]
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: q_num_heads = 32

    return %out#0, %out#1 : tensor<?x?x4096xf16>, tensor<?x8x128x128xf16>
  }

  // Fully dynamic hidden extents must be mapped semantically rather than
  // copied positionally from query/key:
  //   Dk = query_hidden / Hq, Dv = value_hidden / Hkv.
  func.func @test_linear_dynamic_hidden(
      %query: tensor<?x?x?xf16>,
      %key: tensor<?x?x?xf16>,
      %value: tensor<?x?x?xf16>)
      -> (tensor<?x?x?xf16>, tensor<?x8x?x?xf16>) {
    %out:2 = "onnx.Custom"(%query, %key, %value)
        <{function_name = "LinearAttention"}>
        {domain_name = "com.microsoft",
         q_num_heads = 32 : si64,
         kv_num_heads = 8 : si64}
        : (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>)
        -> (tensor<?x?x?xf16>, tensor<?x8x?x?xf16>)

    // CHECK-LABEL: func.func @test_linear_dynamic_hidden
    // CHECK-SAME: %[[Q7:[^,]+]]: tensor<?x?x?xf16>
    // CHECK-SAME: %[[K7:[^,]+]]: tensor<?x?x?xf16>
    // CHECK-SAME: %[[V7:[^)]+]]: tensor<?x?x?xf16>
    // CHECK: %[[B7:.*]] = tensor.dim %[[Q7]]
    // CHECK: %[[S7:.*]] = tensor.dim %[[Q7]]
    // CHECK: %[[QH7:.*]] = tensor.dim %[[Q7]]
    // CHECK: %[[VH7:.*]] = tensor.dim %[[V7]]
    // CHECK: %[[DK7:.*]] = arith.divui %[[QH7]]
    // CHECK: %[[DV7:.*]] = arith.divui %[[VH7]]
    // CHECK: %[[OH7:.*]] = arith.muli %[[DV7]]
    // CHECK: tensor.empty(%[[B7]], %[[S7]], %[[OH7]]) : tensor<?x?x?xf16>
    // CHECK: tensor.empty(%[[B7]], %[[DK7]], %[[DV7]]) : tensor<?x8x?x?xf16>
    // CHECK: hip.linear_attention

    return %out#0, %out#1
        : tensor<?x?x?xf16>, tensor<?x8x?x?xf16>
  }
}
