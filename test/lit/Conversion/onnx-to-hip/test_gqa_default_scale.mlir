// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Lock down the default-scale computation in GqaConversion.cpp. When the
// GroupQueryAttention `scale` attribute is the ORT sentinel 0.0 ("auto-compute
// at runtime"), the conversion derives scale = 1/sqrt(head_size) from query's
// last dim. head_size must be recovered with the layout-correct divisor:
//   * separate Q/K/V: query dim2 = num_heads * head_size
//       -> divide by num_heads
//   * packed QKV:     query dim2 = (num_heads + 2*kv_num_heads) * head_size
//       (key/value absent) -> divide by (num_heads + 2*kv_num_heads)
//
// Regression guard for the packed-QKV bug (ROCm/hip-ep #621): using num_heads
// for the packed layout recovers head_size too large and bakes a wrong scale.
// e.g. H=40, G=10, d=128: query dim2 = (40 + 2*10)*128 = 7680.
//   correct: 7680/60 = 128 -> scale = 1/sqrt(128) ~= 0.0883883
//   old bug: 7680/40 = 192 -> scale = 1/sqrt(192) ~= 0.0721688
// The scale CHECK below fails on the old code and passes on the fixed code.
//
// Both cases omit `scale` (equivalently scale = 0.0) so the default-scale path
// is exercised -- the existing test_gqa.mlir cases all pass an explicit scale
// and therefore never reach this code.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // ===========================================================================
  // Test 1: Packed QKV, default scale (divisor = num_heads + 2*kv_num_heads)
  // H=40, G=10, d=128 -> query dim2 = (40 + 20)*128 = 7680; scale = 1/sqrt(128)
  // ===========================================================================
  // Named @main_graph so --hip-add-context-arg fires (it requires a
  // @main_graph in the module); the second case below is lowered too.
  func.func @main_graph(
      %query: tensor<1x128x7680xf16>,
      %past_key: tensor<1x10x0x128xf16>,
      %past_value: tensor<1x10x0x128xf16>,
      %seqlens_k: tensor<1xi32>,
      %total_seq_len: tensor<i32>)
      -> (tensor<1x128x5120xf16>, tensor<1x10x128x128xf16>, tensor<1x10x128x128xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %none_key = "onnx.NoValue"() {value} : () -> none
    %none_value = "onnx.NoValue"() {value} : () -> none
    %none1 = "onnx.NoValue"() {value} : () -> none
    %none2 = "onnx.NoValue"() {value} : () -> none

    // Packed QKV: query carries all QKV data, key/value are NoValue. `scale`
    // is omitted -> ORT sentinel 0.0 -> conversion computes the default.
    %out:3 = "onnx.Custom"(%query, %none_key, %none_value, %past_key, %past_value,
                            %seqlens_k, %total_seq_len, %none1, %none2)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 40 : si64,
         kv_num_heads = 10 : si64,
         softcap = 0.000000e+00 : f32,
         do_rotary = 0 : si64,
         rotary_interleaved = 0 : si64}
        : (tensor<1x128x7680xf16>, none, none,
           tensor<1x10x0x128xf16>, tensor<1x10x0x128xf16>,
           tensor<1xi32>, tensor<i32>, none, none)
        -> (tensor<1x128x5120xf16>, tensor<1x10x128x128xf16>, tensor<1x10x128x128xf16>)

    // head_size = 7680 / (40 + 2*10) = 128 -> scale = 1/sqrt(128) ~= 0.0883883.
    // Old (buggy) divisor num_heads=40 would give 1/sqrt(192) ~= 0.0721688.
    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: kv_num_heads = 10
    // CHECK-SAME: num_heads = 40
    // CHECK-SAME: scale = {{0.088388[0-9]*|8.838834[0-9]*e-02}}

    return %out#0, %out#1, %out#2 : tensor<1x128x5120xf16>, tensor<1x10x128x128xf16>, tensor<1x10x128x128xf16>
  }

  // ===========================================================================
  // Test 2: Separate Q/K/V, default scale (divisor = num_heads)
  // H=32, G=8, d=128 -> query dim2 = 32*128 = 4096; scale = 1/sqrt(128)
  // Pins the non-packed branch of the divisor selection.
  // ===========================================================================
  func.func @separate_qkv_default_scale(
      %query: tensor<1x1x4096xf16>,
      %key: tensor<1x1x1024xf16>,
      %value: tensor<1x1x1024xf16>,
      %past_key: tensor<1x8x127x128xf16>,
      %past_value: tensor<1x8x127x128xf16>,
      %seqlens_k: tensor<1x1xi32>,
      %total_seq_len: tensor<i32>)
      -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @separate_qkv_default_scale
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %none1 = "onnx.NoValue"() {value} : () -> none
    %none2 = "onnx.NoValue"() {value} : () -> none

    // Separate K/V present; `scale` omitted -> default computed with divisor
    // num_heads (packed factor must NOT be applied here).
    %out:3 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                            %seqlens_k, %total_seq_len, %none1, %none2)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 32 : si64,
         kv_num_heads = 8 : si64,
         softcap = 0.000000e+00 : f32,
         do_rotary = 0 : si64,
         rotary_interleaved = 0 : si64}
        : (tensor<1x1x4096xf16>, tensor<1x1x1024xf16>, tensor<1x1x1024xf16>,
           tensor<1x8x127x128xf16>, tensor<1x8x127x128xf16>,
           tensor<1x1xi32>, tensor<i32>, none, none)
        -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)

    // head_size = 4096 / 32 = 128 -> scale = 1/sqrt(128) ~= 0.0883883.
    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: num_heads = 32
    // CHECK-SAME: scale = {{0.088388[0-9]*|8.838834[0-9]*e-02}}

    return %out#0, %out#1, %out#2 : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
  }
}
