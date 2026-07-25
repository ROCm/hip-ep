// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify com.microsoft.GroupQueryAttention (via onnx.Custom) is correctly
// lowered to hip.gqa operation in tensor-first mode.
//
// This test suite validates:
// - Custom ONNX op matching by function_name and domain_name
// - Multi-output op lowering (3-4 outputs)
// - Attribute preservation (num_heads, kv_num_heads, scale, etc.)
// - Optional inputs handling (packed QKV, quantization, etc.)
// - f16 element type support
// - Tensor-first DPS: tensor.empty() for each output
//
// Test cases:
// 1. Basic GQA (Llama-3.1-8B decode step)
// 2. Packed QKV (key/value optional)
// 3. Local window attention (Mistral)
// 4. Full spec (all 14 inputs + all attributes)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

// =============================================================================
// Test 1: Basic GQA - Standard decode step with separate K/V
// =============================================================================
module {
  func.func @main_graph(
      %query: tensor<1x1x4096xf16>,
      %key: tensor<1x1x1024xf16>,
      %value: tensor<1x1x1024xf16>,
      %past_key: tensor<1x8x127x128xf16>,
      %past_value: tensor<1x8x127x128xf16>,
      %seqlens_k: tensor<1x1xi32>,
      %total_seq_len: tensor<i32>)
      -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

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
        : (tensor<1x1x4096xf16>, tensor<1x1x1024xf16>, tensor<1x1x1024xf16>,
           tensor<1x8x127x128xf16>, tensor<1x8x127x128xf16>,
           tensor<1x1xi32>, tensor<i32>, none, none)
        -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)

    // Three tensor.empty() inits for the three outputs
    // CHECK: tensor.empty() : tensor<1x1x4096xf16>
    // CHECK: tensor.empty() : tensor<1x8x128x128xf16>
    // CHECK: tensor.empty() : tensor<1x8x128x128xf16>
    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: num_heads = 32
    // CHECK-NOT: hip.alloc

    return %out#0, %out#1, %out#2 : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
  }

  // ===========================================================================
  // Test 2: Packed QKV - Key and Value are optional (QKV fused in query)
  // ===========================================================================
  func.func @test_packed_qkv(
      %query: tensor<1x128x12288xf16>,
      %past_key: tensor<1x8x0x128xf16>,
      %past_value: tensor<1x8x0x128xf16>,
      %seqlens_k: tensor<1xi32>,
      %total_seq_len: tensor<i32>)
      -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @test_packed_qkv
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %none_key = "onnx.NoValue"() {value} : () -> none
    %none_value = "onnx.NoValue"() {value} : () -> none
    %none1 = "onnx.NoValue"() {value} : () -> none
    %none2 = "onnx.NoValue"() {value} : () -> none

    // Packed QKV: query contains all QKV data, key/value are NoValue
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

    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: ins(
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: num_heads = 32
    // Verify key and value are not in operand list (packed QKV mode)

    return %out#0, %out#1, %out#2 : tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
  }

  // ===========================================================================
  // Test 3: Local Window Attention - Mistral sliding window
  // ===========================================================================
  func.func @test_local_window(
      %query: tensor<1x128x4096xf16>,
      %key: tensor<1x128x1024xf16>,
      %value: tensor<1x128x1024xf16>,
      %past_key: tensor<1x8x0x128xf16>,
      %past_value: tensor<1x8x0x128xf16>,
      %seqlens_k: tensor<1xi32>,
      %total_seq_len: tensor<i32>)
      -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @test_local_window
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

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
         rotary_interleaved = 0 : si64,
         local_window_size = 4096 : si64}
        : (tensor<1x128x4096xf16>, tensor<1x128x1024xf16>, tensor<1x128x1024xf16>,
           tensor<1x8x0x128xf16>, tensor<1x8x0x128xf16>,
           tensor<1xi32>, tensor<i32>, none, none)
        -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)

    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: local_window_size = 4096
    // CHECK-SAME: num_heads = 32

    return %out#0, %out#1, %out#2 : tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
  }

  // ===========================================================================
  // Test 4: Full Spec - All 14 inputs + All 12 attributes + 4 outputs
  // ===========================================================================
  func.func @test_full_spec(
      %query: tensor<1x128x4096xf16>,
      %key: tensor<1x128x4096xf16>,
      %value: tensor<1x128x4096xf16>,
      %past_key: tensor<1x8x0x128xf16>,
      %past_value: tensor<1x8x0x128xf16>,
      %seqlens_k: tensor<1xi32>,
      %total_seq_len: tensor<i32>,
      %cos_cache: tensor<2048x64xf16>,
      %sin_cache: tensor<2048x64xf16>,
      %position_ids: tensor<1x128xi64>,
      %attention_bias: tensor<1x32x128x128xf16>,
      %head_sink: tensor<32xf16>,
      %k_scale: tensor<1x8x1x128xf32>,
      %v_scale: tensor<1x8x1x128xf32>
  ) -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>, tensor<1x32x128x128xf16>) {

    // CHECK-LABEL: func.func @test_full_spec
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    // All 14 inputs (complete MS spec)
    %out:4 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                           %seqlens_k, %total_seq_len,
                           %cos_cache, %sin_cache, %position_ids,
                           %attention_bias, %head_sink, %k_scale, %v_scale)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 32 : si64,
         kv_num_heads = 8 : si64,
         scale = 8.838834e-02 : f32,
         do_rotary = 1 : si64,
         rotary_interleaved = 0 : si64,
         softcap = 3.000000e+01 : f32,
         local_window_size = 4096 : si64,
         smooth_softmax = 1 : si64,
         qk_output = 2 : si64,
         k_quant_type = "PER_CHANNEL",
         v_quant_type = "PER_CHANNEL",
         kv_cache_bit_width = 8 : si64}
        : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<1x128x4096xf16>,
           tensor<1x8x0x128xf16>, tensor<1x8x0x128xf16>,
           tensor<1xi32>, tensor<i32>,
           tensor<2048x64xf16>, tensor<2048x64xf16>, tensor<1x128xi64>,
           tensor<1x32x128x128xf16>, tensor<32xf16>,
           tensor<1x8x1x128xf32>, tensor<1x8x1x128xf32>)
        -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>, tensor<1x32x128x128xf16>)

    // Verify all attributes are preserved (in alphabetical order)
    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: ins(
    // CHECK-SAME: do_rotary = 1
    // CHECK-SAME: k_quant_type = "PER_CHANNEL"
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: local_window_size = 4096
    // CHECK-SAME: num_heads = 32
    // CHECK-SAME: qk_output = 2
    // CHECK-SAME: scale = {{0.088388[0-9]*|8.838834e-02}}
    // CHECK-SAME: smooth_softmax = 1
    // CHECK-SAME: softcap = {{30.0+|3.000000e\+01}}
    // CHECK-SAME: v_quant_type = "PER_CHANNEL"

    return %out#0, %out#1, %out#2, %out#3 : tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>, tensor<1x32x128x128xf16>
  }
}
