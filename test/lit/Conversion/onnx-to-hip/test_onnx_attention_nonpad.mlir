// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify onnx.Attention with nonpad_kv_seqlen and is_causal = 1 lowers to
// hip.kv_attention: K/V are read in place from a buffer whose first
// nonpad_kv_seqlen[b] rows are valid, and nothing is appended.
//   - rank-3 K/V [B, T, G * d] are split into heads and transposed to BNSH,
//   - rank-4 K/V pass through; a rank-4 Q is flattened to [B, S, H * d] and
//     the output mapped back,
//   - without a scale attribute the scale is 1 / sqrt(d).
// Each negative case keeps the onnx.Attention, since hip.kv_attention can
// not express it:
//   - is_causal = 0,
//   - an attn_mask next to nonpad_kv_seqlen,
//   - a past_key / past_value with used present outputs.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --split-input-file | FileCheck %s

// ===== rank-3 K/V, 8 query heads over 2 KV heads, default scale =====
module {
  func.func @main_graph(%q: tensor<1x?x1024xf16>, %k: tensor<1x256x256xf16>,
                        %v: tensor<1x256x256xf16>, %len: tensor<1xi64>)
      -> tensor<1x?x1024xf16> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[Q:[^:]*]]: tensor<1x?x1024xf16>, %[[K:[^:]*]]: tensor<1x256x256xf16>, %[[V:[^:]*]]: tensor<1x256x256xf16>, %[[LEN:[^:]*]]: tensor<1xi64>)
    // CHECK: %[[K4:.*]] = tensor.expand_shape %[[K]] {{\[\[}}0], [1], [2, 3]] output_shape [1, 256, 2, 128]
    // CHECK: %[[KT:.*]] = hip.transpose(%[[CTX]]) ins(%[[K4]] : tensor<1x256x2x128xf16>) outs(%{{.*}} : tensor<1x2x256x128xf16>) {perm = [0, 2, 1, 3]}
    // CHECK: %[[V4:.*]] = tensor.expand_shape %[[V]]
    // CHECK: %[[VT:.*]] = hip.transpose(%[[CTX]]) ins(%[[V4]] : tensor<1x256x2x128xf16>)
    // CHECK: %[[Y:.*]] = hip.kv_attention(%[[CTX]]) ins(%[[Q]], %[[KT]], %[[VT]] : tensor<1x?x1024xf16>, tensor<1x2x256x128xf16>, tensor<1x2x256x128xf16>) kv_seqlen(%[[LEN]] : tensor<1xi64>)
    // CHECK-SAME: {kv_num_heads = 2 : i64, num_heads = 8 : i64, scale = 0.0883883461 : f32}
    // CHECK-NOT: hip.gqa
    // CHECK: return %[[Y]]
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %len)
        {is_causal = 1 : si64, kv_num_heads = 2 : si64, q_num_heads = 8 : si64}
        : (tensor<1x?x1024xf16>, tensor<1x256x256xf16>, tensor<1x256x256xf16>, none, none, none, tensor<1xi64>)
        -> tensor<1x?x1024xf16>
    return %y : tensor<1x?x1024xf16>
  }
}

// -----

// ===== rank-4 Q and BNSH K/V =====
module {
  func.func @main_graph(%q: tensor<1x8x?x128xf16>, %k: tensor<1x2x256x128xf16>,
                        %v: tensor<1x2x256x128xf16>, %len: tensor<1xi64>)
      -> tensor<1x8x?x128xf16> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[Q:[^:]*]]: tensor<1x8x?x128xf16>, %[[K:[^:]*]]: tensor<1x2x256x128xf16>, %[[V:[^:]*]]: tensor<1x2x256x128xf16>, %[[LEN:[^:]*]]: tensor<1xi64>)
    // CHECK: %[[QT:.*]] = hip.transpose(%[[CTX]]) ins(%[[Q]] : tensor<1x8x?x128xf16>) outs(%{{.*}} : tensor<1x?x8x128xf16>) {perm = [0, 2, 1, 3]}
    // CHECK: %[[Q3:.*]] = tensor.collapse_shape %[[QT]] {{\[\[}}0], [1], [2, 3]] : tensor<1x?x8x128xf16> into tensor<1x?x1024xf16>
    // CHECK: %[[Y:.*]] = hip.kv_attention(%[[CTX]]) ins(%[[Q3]], %[[K]], %[[V]] : tensor<1x?x1024xf16>, tensor<1x2x256x128xf16>, tensor<1x2x256x128xf16>) kv_seqlen(%[[LEN]] : tensor<1xi64>)
    // CHECK-SAME: {kv_num_heads = 2 : i64, num_heads = 8 : i64, scale = 1.250000e-01 : f32}
    // CHECK: %[[Y4:.*]] = tensor.expand_shape %[[Y]] {{\[\[}}0], [1], [2, 3]]
    // CHECK: %[[OUT:.*]] = hip.transpose(%[[CTX]]) ins(%[[Y4]] : tensor<1x?x8x128xf16>) outs(%{{.*}} : tensor<1x8x?x128xf16>) {perm = [0, 2, 1, 3]}
    // CHECK: return %[[OUT]]
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %len)
        {is_causal = 1 : si64, scale = 1.250000e-01 : f32}
        : (tensor<1x8x?x128xf16>, tensor<1x2x256x128xf16>, tensor<1x2x256x128xf16>, none, none, none, tensor<1xi64>)
        -> tensor<1x8x?x128xf16>
    return %y : tensor<1x8x?x128xf16>
  }
}

// -----

// ===== Negative: is_causal = 0 =====
module {
  func.func @main_graph(%q: tensor<1x?x1024xf16>, %k: tensor<1x256x256xf16>,
                        %v: tensor<1x256x256xf16>, %len: tensor<1xi64>)
      -> tensor<1x?x1024xf16> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: hip.kv_attention
    // CHECK: onnx.Attention
    // CHECK-NOT: hip.kv_attention
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %len)
        {kv_num_heads = 2 : si64, q_num_heads = 8 : si64}
        : (tensor<1x?x1024xf16>, tensor<1x256x256xf16>, tensor<1x256x256xf16>, none, none, none, tensor<1xi64>)
        -> tensor<1x?x1024xf16>
    return %y : tensor<1x?x1024xf16>
  }
}

// -----

// ===== Negative: an attn_mask next to nonpad_kv_seqlen =====
module {
  func.func @main_graph(%q: tensor<1x?x1024xf16>, %k: tensor<1x256x256xf16>,
                        %v: tensor<1x256x256xf16>, %mask: tensor<1x1x?x256xf16>,
                        %len: tensor<1xi64>)
      -> tensor<1x?x1024xf16> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: hip.kv_attention
    // CHECK: onnx.Attention
    // CHECK-NOT: hip.kv_attention
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %mask, %none, %none, %len)
        {is_causal = 1 : si64, kv_num_heads = 2 : si64, q_num_heads = 8 : si64}
        : (tensor<1x?x1024xf16>, tensor<1x256x256xf16>, tensor<1x256x256xf16>, tensor<1x1x?x256xf16>, none, none, tensor<1xi64>)
        -> tensor<1x?x1024xf16>
    return %y : tensor<1x?x1024xf16>
  }
}

// -----

// ===== Negative: past K/V with the present outputs returned =====
module {
  func.func @main_graph(%q: tensor<1x?x1024xf16>, %k: tensor<1x?x256xf16>,
                        %v: tensor<1x?x256xf16>, %pk: tensor<1x2x?x128xf16>,
                        %pv: tensor<1x2x?x128xf16>, %len: tensor<1xi64>)
      -> (tensor<1x?x1024xf16>, tensor<1x2x?x128xf16>, tensor<1x2x?x128xf16>) {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: hip.kv_attention
    // CHECK: onnx.Attention
    // CHECK-NOT: hip.kv_attention
    %none = "onnx.NoValue"() {value} : () -> none
    %y, %prk, %prv = "onnx.Attention"(%q, %k, %v, %none, %pk, %pv, %len)
        {is_causal = 1 : si64, kv_num_heads = 2 : si64, q_num_heads = 8 : si64}
        : (tensor<1x?x1024xf16>, tensor<1x?x256xf16>, tensor<1x?x256xf16>, none, tensor<1x2x?x128xf16>, tensor<1x2x?x128xf16>, tensor<1xi64>)
        -> (tensor<1x?x1024xf16>, tensor<1x2x?x128xf16>, tensor<1x2x?x128xf16>)
    return %y, %prk, %prv : tensor<1x?x1024xf16>, tensor<1x2x?x128xf16>, tensor<1x2x?x128xf16>
  }
}
