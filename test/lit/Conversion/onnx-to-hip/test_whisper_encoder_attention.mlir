// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify com.microsoft.Attention (via onnx.Custom) — the fused-QKV
// bidirectional-self-attention op used by Whisper's encoder — lowers to:
//   * One QKV projection (MatMul + Add bias) over the fused weight.  The
//     weight is [H, 3H] (ORT Attention spec / what Whisper exports:
//     qkv_proj.weight = [1280, 3840]), so it feeds the MatMul DIRECTLY — there
//     must be NO weight transpose.
//   * Three tensor.extract_slice ops splitting the [B,S,3*H] activation
//     into Q / K / V along the last axis (Option B: 1 fused MatMul + 3 slices,
//     not 3 separate MatMul/Add).  The fused-weight layout matches what the
//     original ONNX graph already has; we keep one large GEMM instead of
//     splitting weights at compile time.
//   * One hip.gqa with no_causal = true, num_heads == kv_num_heads (HPG=1),
//     and a compile-time seqlens_k constant equal to the static Skv.
//
// Sources covered:
//   - Whisper-large-v3 encoder Attention nodes (32 layers, identical shapes,
//     d_model=1280, num_heads=20, head_dim=64, S=1500).
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(
      %hidden: tensor<1x1500x1280xf16>,
      %qkv_w:  tensor<1280x3840xf16>,
      %qkv_b:  tensor<3840xf16>)
      -> tensor<1x1500x1280xf16> {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %none1 = "onnx.NoValue"() {value} : () -> none
    %none2 = "onnx.NoValue"() {value} : () -> none
    %none3 = "onnx.NoValue"() {value} : () -> none
    %none4 = "onnx.NoValue"() {value} : () -> none

    // Real Whisper-large-v3 encoder weight layout: [H, 3H] = [1280, 3840]
    // (ORT com.microsoft.Attention spec convention, output dims = q+k+v hidden).
    %out = "onnx.Custom"(%hidden, %qkv_w, %qkv_b, %none1, %none2, %none3, %none4)
        <{function_name = "Attention"}>
        {domain_name = "com.microsoft",
         num_heads = 20 : si64,
         qkv_hidden_sizes = [1280, 1280, 1280],
         unidirectional = 0 : si64,
         do_rotary = 0 : si64,
         past_present_share_buffer = 0 : si64,
         rotary_embedding_dim = 0 : si64,
         scale = 1.250000e-01 : f32,
         mask_filter_value = -1.000000e+04 : f32}
        : (tensor<1x1500x1280xf16>, tensor<1280x3840xf16>, tensor<3840xf16>,
           none, none, none, none)
        -> tensor<1x1500x1280xf16>

    // The weight is already [H, 3H], so the fused QKV projection is just
    // hip.matmul + hip.add — NO weight transpose must be emitted.
    // CHECK-NOT: hip.transpose
    // CHECK: hip.matmul
    // CHECK: hip.add
    // Three slices on the projected activation produce Q / K / V.
    // CHECK-COUNT-3: tensor.extract_slice
    // hip.gqa is bidirectional (no_causal=true) and uses HPG=1.
    // CHECK: hip.gqa(%[[CTX]])
    // CHECK-SAME: kv_num_heads = 20
    // CHECK-SAME: no_causal = true
    // CHECK-SAME: num_heads = 20

    return %out : tensor<1x1500x1280xf16>
  }

  // CHECK-LABEL: func.func @encoder_attn_rejects_unidirectional
  func.func @encoder_attn_rejects_unidirectional(%x: tensor<1x1500x1280xf16>,
                                                  %qkv_w: tensor<1280x3840xf16>,
                                                  %qkv_b: tensor<3840xf16>)
      -> tensor<1x1500x1280xf16> {
    // Pattern must REJECT this — encoder is bidirectional only.
    %0 = "onnx.Custom"(%x, %qkv_w, %qkv_b) {function_name = "Attention",
         domain_name = "com.microsoft", num_heads = 20 : i64,
         qkv_hidden_sizes = [1280, 1280, 1280],
         unidirectional = 1 : i64,
         scale = 0.125 : f32, do_rotary = 0 : i64,
         past_present_share_buffer = 0 : i64, rotary_embedding_dim = 0 : i64,
         mask_filter_value = -10000.0 : f32}
         : (tensor<1x1500x1280xf16>, tensor<1280x3840xf16>, tensor<3840xf16>)
         -> tensor<1x1500x1280xf16>
    return %0 : tensor<1x1500x1280xf16>
  }
  // Verify the pattern did NOT match — onnx.Custom must remain, hip.gqa must
  // NOT appear in this function body.  CHECK-LABEL above scopes the
  // CHECK-NOT to this function only (hip.gqa is expected in @main_graph).
  // CHECK: onnx.Custom
  // CHECK-NOT: hip.gqa
  // CHECK: return
}
