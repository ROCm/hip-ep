// RUN: hip-mlir-opt %s --convert-onnx-to-hip | FileCheck %s --check-prefix=HIP

// =============================================================================
// Test Suite: GQA Full MS Specification Support
// =============================================================================
//
// This file tests ONNX→HIP conversion for GroupQueryAttention with all
// combinations of optional inputs and attributes according to the complete
// Microsoft ONNX Runtime specification.
//
// Test coverage:
// 1. Minimal GQA (7 inputs - standard configuration)
// 2. GQA with RoPE (cos_cache, sin_cache, position_ids)
// 3. GQA with attention_bias (ALiBi support)
// 4. GQA with quantization (k_scale, v_scale, quant_type attributes)
// 5. GQA with local window (Mistral sliding window attention)
// 6. GQA full spec (all 14 inputs + all 12 attributes)
// =============================================================================

// -----------------------------------------------------------------------------
// Test Case 1: Minimal GQA (7 inputs - standard configuration)
// -----------------------------------------------------------------------------
// This represents the most common usage: query, key, value, past caches,
// sequence length info. No optional features.

func.func @test_gqa_minimal(
    %query: tensor<1x128x4096xf16>,
    %key: tensor<1x128x4096xf16>,
    %value: tensor<1x128x4096xf16>,
    %past_key: tensor<1x8x0x128xf16>,
    %past_value: tensor<1x8x0x128xf16>,
    %seqlens_k: tensor<1xi32>,
    %total_seq_len: tensor<i32>
) -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>) {

  %0:3 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len) {
    function_name = "GroupQueryAttention",
    domain_name = "com.microsoft",
    num_heads = 32 : si64,
    kv_num_heads = 8 : si64
  } : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<1x128x4096xf16>,
       tensor<1x8x0x128xf16>, tensor<1x8x0x128xf16>,
       tensor<1xi32>, tensor<i32>)
    -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)

  return %0#0, %0#1, %0#2 : tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
}

// HIP-LABEL: func.func @test_gqa_minimal
// HIP: hip.gqa
// HIP-SAME: num_heads = 32
// HIP-SAME: kv_num_heads = 8
// HIP-SAME: scale = 1.000000e+00
// HIP-SAME: do_rotary = 0
// HIP-SAME: rotary_interleaved = 0
// HIP-SAME: softcap = 0.000000e+00
// HIP-SAME: local_window_size = -1
// HIP-SAME: smooth_softmax = 0
// HIP-SAME: qk_output = 0
// HIP-SAME: k_quant_type = "NONE"
// HIP-SAME: v_quant_type = "NONE"
// HIP-SAME: kv_cache_bit_width = 8

// -----------------------------------------------------------------------------
// Test Case 2: GQA with RoPE (cos_cache, sin_cache, position_ids)
// -----------------------------------------------------------------------------

func.func @test_gqa_with_rope(
    %query: tensor<1x128x4096xf16>,
    %key: tensor<1x128x4096xf16>,
    %value: tensor<1x128x4096xf16>,
    %past_key: tensor<1x8x0x128xf16>,
    %past_value: tensor<1x8x0x128xf16>,
    %seqlens_k: tensor<1xi32>,
    %total_seq_len: tensor<i32>,
    %cos_cache: tensor<2048x64xf16>,
    %sin_cache: tensor<2048x64xf16>,
    %position_ids: tensor<1x128xi64>
) -> tensor<1x128x4096xf16> {

  %0:3 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                       %seqlens_k, %total_seq_len,
                       %cos_cache, %sin_cache, %position_ids) {
    function_name = "GroupQueryAttention",
    domain_name = "com.microsoft",
    num_heads = 32 : si64,
    kv_num_heads = 8 : si64,
    do_rotary = 1 : si64,
    rotary_interleaved = 0 : si64
  } : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<1x128x4096xf16>,
       tensor<1x8x0x128xf16>, tensor<1x8x0x128xf16>,
       tensor<1xi32>, tensor<i32>,
       tensor<2048x64xf16>, tensor<2048x64xf16>, tensor<1x128xi64>)
    -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)

  return %0#0 : tensor<1x128x4096xf16>
}

// HIP-LABEL: func.func @test_gqa_with_rope
// HIP: hip.gqa
// HIP-SAME: do_rotary = 1
// HIP-SAME: rotary_interleaved = 0

// -----------------------------------------------------------------------------
// Test Case 3: GQA with attention_bias (ALiBi support)
// -----------------------------------------------------------------------------

func.func @test_gqa_with_bias(
    %query: tensor<1x128x4096xf16>,
    %key: tensor<1x128x4096xf16>,
    %value: tensor<1x128x4096xf16>,
    %past_key: tensor<1x8x0x128xf16>,
    %past_value: tensor<1x8x0x128xf16>,
    %seqlens_k: tensor<1xi32>,
    %total_seq_len: tensor<i32>,
    %none_1: none,
    %none_2: none,
    %none_3: none,
    %attention_bias: tensor<1x32x128x128xf16>
) -> tensor<1x128x4096xf16> {

  %0:3 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                       %seqlens_k, %total_seq_len,
                       %none_1, %none_2, %none_3,
                       %attention_bias) {
    function_name = "GroupQueryAttention",
    domain_name = "com.microsoft",
    num_heads = 32 : si64,
    kv_num_heads = 8 : si64
  } : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<1x128x4096xf16>,
       tensor<1x8x0x128xf16>, tensor<1x8x0x128xf16>,
       tensor<1xi32>, tensor<i32>,
       none, none, none, tensor<1x32x128x128xf16>)
    -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)

  return %0#0 : tensor<1x128x4096xf16>
}

// HIP-LABEL: func.func @test_gqa_with_bias
// HIP: hip.gqa

// -----------------------------------------------------------------------------
// Test Case 4: GQA with quantization (k_scale, v_scale)
// -----------------------------------------------------------------------------

func.func @test_gqa_quantized(
    %query: tensor<1x128x4096xf16>,
    %key: tensor<1x128x4096xf16>,
    %value: tensor<1x128x4096xf16>,
    %past_key: tensor<1x8x0x128xf16>,
    %past_value: tensor<1x8x0x128xf16>,
    %seqlens_k: tensor<1xi32>,
    %total_seq_len: tensor<i32>,
    %none_1: none,
    %none_2: none,
    %none_3: none,
    %none_4: none,
    %none_5: none,
    %k_scale: tensor<f32>,
    %v_scale: tensor<f32>
) -> tensor<1x128x4096xf16> {

  %0:3 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                       %seqlens_k, %total_seq_len,
                       %none_1, %none_2, %none_3, %none_4, %none_5,
                       %k_scale, %v_scale) {
    function_name = "GroupQueryAttention",
    domain_name = "com.microsoft",
    num_heads = 32 : si64,
    kv_num_heads = 8 : si64,
    k_quant_type = "PER_TENSOR",
    v_quant_type = "PER_TENSOR",
    kv_cache_bit_width = 8 : si64
  } : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<1x128x4096xf16>,
       tensor<1x8x0x128xf16>, tensor<1x8x0x128xf16>,
       tensor<1xi32>, tensor<i32>,
       none, none, none, none, none,
       tensor<f32>, tensor<f32>)
    -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)

  return %0#0 : tensor<1x128x4096xf16>
}

// HIP-LABEL: func.func @test_gqa_quantized
// HIP: hip.gqa
// HIP-SAME: k_quant_type = "PER_TENSOR"
// HIP-SAME: v_quant_type = "PER_TENSOR"
// HIP-SAME: kv_cache_bit_width = 8

// -----------------------------------------------------------------------------
// Test Case 5: GQA with local window (Mistral sliding window attention)
// -----------------------------------------------------------------------------

func.func @test_gqa_local_window(
    %query: tensor<1x128x4096xf16>,
    %key: tensor<1x128x4096xf16>,
    %value: tensor<1x128x4096xf16>,
    %past_key: tensor<1x8x0x128xf16>,
    %past_value: tensor<1x8x0x128xf16>,
    %seqlens_k: tensor<1xi32>,
    %total_seq_len: tensor<i32>
) -> tensor<1x128x4096xf16> {

  %0:3 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                       %seqlens_k, %total_seq_len) {
    function_name = "GroupQueryAttention",
    domain_name = "com.microsoft",
    num_heads = 32 : si64,
    kv_num_heads = 8 : si64,
    local_window_size = 4096 : si64
  } : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<1x128x4096xf16>,
       tensor<1x8x0x128xf16>, tensor<1x8x0x128xf16>,
       tensor<1xi32>, tensor<i32>)
    -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)

  return %0#0 : tensor<1x128x4096xf16>
}

// HIP-LABEL: func.func @test_gqa_local_window
// HIP: hip.gqa
// HIP-SAME: local_window_size = 4096

// -----------------------------------------------------------------------------
// Test Case 6: GQA Full Spec (all 14 inputs + all 12 attributes)
// -----------------------------------------------------------------------------

func.func @test_gqa_full_spec(
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

  %0:4 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                       %seqlens_k, %total_seq_len,
                       %cos_cache, %sin_cache, %position_ids,
                       %attention_bias, %head_sink, %k_scale, %v_scale) {
    function_name = "GroupQueryAttention",
    domain_name = "com.microsoft",
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
    kv_cache_bit_width = 8 : si64
  } : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<1x128x4096xf16>,
       tensor<1x8x0x128xf16>, tensor<1x8x0x128xf16>,
       tensor<1xi32>, tensor<i32>,
       tensor<2048x64xf16>, tensor<2048x64xf16>, tensor<1x128xi64>,
       tensor<1x32x128x128xf16>, tensor<32xf16>,
       tensor<1x8x1x128xf32>, tensor<1x8x1x128xf32>)
    -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>, tensor<1x32x128x128xf16>)

  return %0#0, %0#1, %0#2, %0#3 : tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>, tensor<1x32x128x128xf16>
}

// HIP-LABEL: func.func @test_gqa_full_spec
// HIP: hip.gqa
// HIP-SAME: num_heads = 32
// HIP-SAME: kv_num_heads = 8
// HIP-SAME: scale = 8.838834e-02
// HIP-SAME: do_rotary = 1
// HIP-SAME: rotary_interleaved = 0
// HIP-SAME: softcap = 3.000000e+01
// HIP-SAME: local_window_size = 4096
// HIP-SAME: smooth_softmax = 1
// HIP-SAME: qk_output = 2
// HIP-SAME: k_quant_type = "PER_CHANNEL"
// HIP-SAME: v_quant_type = "PER_CHANNEL"
// HIP-SAME: kv_cache_bit_width = 8
