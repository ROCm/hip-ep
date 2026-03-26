// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

// =============================================================================
// Test: GQA Full MS Specification (All 14 Inputs + All 12 Attributes)
// =============================================================================
//
// This test validates ONNX→HIP conversion for GroupQueryAttention with the
// complete Microsoft ONNX Runtime specification:
// - All 14 inputs (including all optional features)
// - All 12 attributes (including quantization, local window, smooth softmax, etc.)
// - 4 outputs (including optional output_qk for debugging)
//
// Tests the most comprehensive GQA configuration possible.
// =============================================================================

module {
  func.func @main_graph(
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

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

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
}

// Verify hip.gqa operation is generated with all inputs and attributes
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
