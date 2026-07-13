// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify native onnx.Attention (ONNX opset 23/24) lowers to hip.gqa with:
//   - external attn_mask threaded as attention_bias
//   - is_causal=0 -> no_causal=true (built-in causal mask skipped)
//   - runtime seqlens_k derived from past_key seq dim
//   - present_key/value DPS inits
//
// Shape pattern mirrors Gemma-style GQA decode (16 query heads, 8 KV heads).
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

    // CHECK: tensor.dim %{{.*}}, %c2
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
}
