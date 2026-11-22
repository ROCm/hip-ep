// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @correction_bias_with_softmax(%ctx: !hip.context,
    %input: memref<1x8x256xf16, 1>, %logits: memref<1x8x16xf32, 1>,
    %fc1_w: memref<16x256x128xui8, 1>, %fc1_s: memref<16x256x2xf16, 1>,
    %fc2_w: memref<16x256x64xui8, 1>, %fc2_s: memref<16x256x1xf16, 1>,
    %bias: memref<16xf32, 1>, %output: memref<1x8x256xf16, 1>) {
  // expected-error @+1 {{correction_bias requires routing_type sigmoid_bias}}
  hip.qmoe(%ctx) ins(%input, %logits, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      memref<1x8x256xf16, 1>, memref<1x8x16xf32, 1>,
      memref<16x256x128xui8, 1>, memref<16x256x2xf16, 1>,
      memref<16x256x64xui8, 1>, memref<16x256x1xf16, 1>)
      correction_bias(%bias : memref<16xf32, 1>)
      outs(%output : memref<1x8x256xf16, 1>)
      {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 128 : i64,
       normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64, activation_alpha = 1.0 : f32,
       activation_beta = 0.0 : f32, swiglu_limit = 7.0 : f32,
       activation_type = "swiglu"}
  return
}

// -----

func.func @unknown_routing_type(%ctx: !hip.context,
    %input: memref<1x8x256xf16, 1>, %logits: memref<1x8x16xf32, 1>,
    %fc1_w: memref<16x256x128xui8, 1>, %fc1_s: memref<16x256x2xf16, 1>,
    %fc2_w: memref<16x256x64xui8, 1>, %fc2_s: memref<16x256x1xf16, 1>,
    %bias: memref<16xf32, 1>, %output: memref<1x8x256xf16, 1>) {
  // expected-error @+1 {{routing_type must be softmax or sigmoid_bias, got sigmoid}}
  hip.qmoe(%ctx) ins(%input, %logits, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      memref<1x8x256xf16, 1>, memref<1x8x16xf32, 1>,
      memref<16x256x128xui8, 1>, memref<16x256x2xf16, 1>,
      memref<16x256x64xui8, 1>, memref<16x256x1xf16, 1>)
      correction_bias(%bias : memref<16xf32, 1>)
      outs(%output : memref<1x8x256xf16, 1>)
      {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 128 : i64,
       normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64, activation_alpha = 1.0 : f32,
       activation_beta = 0.0 : f32, swiglu_limit = 7.0 : f32,
       activation_type = "swiglu", routing_type = "sigmoid"}
  return
}

// -----

func.func @sigmoid_bias_with_router_weights(%ctx: !hip.context,
    %input: memref<1x8x256xf16, 1>, %logits: memref<1x8x16xf32, 1>,
    %fc1_w: memref<16x256x128xui8, 1>, %fc1_s: memref<16x256x2xf16, 1>,
    %fc2_w: memref<16x256x64xui8, 1>, %fc2_s: memref<16x256x1xf16, 1>,
    %weights: memref<8x16xf16, 1>, %bias: memref<16xf32, 1>,
    %output: memref<1x8x256xf16, 1>) {
  // expected-error @+1 {{sigmoid_bias routing computes its own weights; router_weights must be absent}}
  hip.qmoe(%ctx) ins(%input, %logits, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      memref<1x8x256xf16, 1>, memref<1x8x16xf32, 1>,
      memref<16x256x128xui8, 1>, memref<16x256x2xf16, 1>,
      memref<16x256x64xui8, 1>, memref<16x256x1xf16, 1>)
      router_weights(%weights : memref<8x16xf16, 1>)
      correction_bias(%bias : memref<16xf32, 1>)
      outs(%output : memref<1x8x256xf16, 1>)
      {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 128 : i64,
       normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64, activation_alpha = 1.0 : f32,
       activation_beta = 0.0 : f32, swiglu_limit = 7.0 : f32,
       activation_type = "swiglu", routing_type = "sigmoid_bias"}
  return
}

// -----

func.func @sigmoid_bias_fp16_logits(%ctx: !hip.context,
    %input: memref<1x8x256xf16, 1>, %logits: memref<1x8x16xf16, 1>,
    %fc1_w: memref<16x256x128xui8, 1>, %fc1_s: memref<16x256x2xf16, 1>,
    %fc2_w: memref<16x256x64xui8, 1>, %fc2_s: memref<16x256x1xf16, 1>,
    %bias: memref<16xf32, 1>, %output: memref<1x8x256xf16, 1>) {
  // expected-error @+1 {{sigmoid_bias routing needs fp32 router logits}}
  hip.qmoe(%ctx) ins(%input, %logits, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      memref<1x8x256xf16, 1>, memref<1x8x16xf16, 1>,
      memref<16x256x128xui8, 1>, memref<16x256x2xf16, 1>,
      memref<16x256x64xui8, 1>, memref<16x256x1xf16, 1>)
      correction_bias(%bias : memref<16xf32, 1>)
      outs(%output : memref<1x8x256xf16, 1>)
      {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 128 : i64,
       normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64, activation_alpha = 1.0 : f32,
       activation_beta = 0.0 : f32, swiglu_limit = 7.0 : f32,
       activation_type = "swiglu", routing_type = "sigmoid_bias"}
  return
}

// -----

func.func @sigmoid_bias_without_correction_bias(%ctx: !hip.context,
    %input: memref<1x8x256xf16, 1>, %logits: memref<1x8x16xf32, 1>,
    %fc1_w: memref<16x256x128xui8, 1>, %fc1_s: memref<16x256x2xf16, 1>,
    %fc2_w: memref<16x256x64xui8, 1>, %fc2_s: memref<16x256x1xf16, 1>,
    %output: memref<1x8x256xf16, 1>) {
  // expected-error @+1 {{sigmoid_bias routing needs a rank-1 fp32 correction_bias}}
  hip.qmoe(%ctx) ins(%input, %logits, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      memref<1x8x256xf16, 1>, memref<1x8x16xf32, 1>,
      memref<16x256x128xui8, 1>, memref<16x256x2xf16, 1>,
      memref<16x256x64xui8, 1>, memref<16x256x1xf16, 1>)
      outs(%output : memref<1x8x256xf16, 1>)
      {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 128 : i64,
       normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64, activation_alpha = 1.0 : f32,
       activation_beta = 0.0 : f32, swiglu_limit = 7.0 : f32,
       activation_type = "swiglu", routing_type = "sigmoid_bias"}
  return
}

// -----

func.func @correction_bias_length_mismatch(%ctx: !hip.context,
    %input: memref<1x8x256xf16, 1>, %logits: memref<1x8x16xf32, 1>,
    %fc1_w: memref<16x256x128xui8, 1>, %fc1_s: memref<16x256x2xf16, 1>,
    %fc2_w: memref<16x256x64xui8, 1>, %fc2_s: memref<16x256x1xf16, 1>,
    %bias: memref<8xf32, 1>, %output: memref<1x8x256xf16, 1>) {
  // expected-error @+1 {{correction_bias length 8 does not match 16 experts}}
  hip.qmoe(%ctx) ins(%input, %logits, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      memref<1x8x256xf16, 1>, memref<1x8x16xf32, 1>,
      memref<16x256x128xui8, 1>, memref<16x256x2xf16, 1>,
      memref<16x256x64xui8, 1>, memref<16x256x1xf16, 1>)
      correction_bias(%bias : memref<8xf32, 1>)
      outs(%output : memref<1x8x256xf16, 1>)
      {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 128 : i64,
       normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64, activation_alpha = 1.0 : f32,
       activation_beta = 0.0 : f32, swiglu_limit = 7.0 : f32,
       activation_type = "swiglu", routing_type = "sigmoid_bias"}
  return
}
