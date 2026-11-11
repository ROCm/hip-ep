// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

// The conversion requires a @main_graph entry function.
func.func @main_graph(%arg0: tensor<1xf16>) -> tensor<1xf16> {
  return %arg0 : tensor<1xf16>
}

// The exported Split, the per-head norm + MRoPE chains of q and k and the
// sigmoid gate fold into one attention op over the packed projection.
// CHECK-LABEL: func.func @fold_all(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QKV:[^:]*]]: tensor<1x2x768xf16>,
// CHECK-SAME:      %[[G:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[POS:[^:]*]]: tensor<3x1x2xi64>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<64x16xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<64x16xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[IDX:[^:]*]]: tensor<2x8xi32>,
// CHECK-SAME:      %[[CNT:[^:]*]]: tensor<2xi32>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
// CHECK-NEXT:      %[[QW:.*]] = hip.constant {serialization_order = 4 : i64, value = dense<1.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[KW:.*]] = hip.constant {serialization_order = 5 : i64, value = dense<2.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[R:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[QKV]] : tensor<1x2x768xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[IDX]], %[[CNT]], %[[SEQLENS]] : tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          gate(%[[G]] : tensor<1x2x512xf16>)
// CHECK-SAME:          prologue(%[[QW]], %[[KW]], %[[POS]], %[[COS]], %[[SIN]] : tensor<128xf16>, tensor<128xf16>, tensor<3x1x2xi64>, tensor<64x16xf16>, tensor<64x16xf16>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, mrope_layout = 1 : i64, mrope_section = array<i64: 6, 5, 5>, norm_epsilon = 9.99999997E-7 : f32, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#1, %[[R]]#2 : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:    }
func.func @fold_all(
    %qkv: tensor<1x2x768xf16>, %g: tensor<1x2x512xf16>, %pos: tensor<3x1x2xi64>,
    %cos: tensor<64x16xf16>, %sin: tensor<64x16xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %idx: tensor<2x8xi32>, %cnt: tensor<2xi32>, %seqlens: tensor<1xi32>,
    %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = "onnx.Constant"() {value = dense<[512, 128, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
  %heads = "onnx.Constant"() {value = dense<[0, -1, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
  %qshape = "onnx.Constant"() {value = dense<[0, -1, 512]> : tensor<3xi64>} : () -> tensor<3xi64>
  %kshape = "onnx.Constant"() {value = dense<[0, -1, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
  %qw = "onnx.Constant"() {value = dense<1.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %kw = "onnx.Constant"() {value = dense<2.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %q, %k, %v = "onnx.Split"(%qkv, %sizes) {axis = -1 : si64} : (tensor<1x2x768xf16>, tensor<3xi64>) -> (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>)
  %qh = "onnx.Reshape"(%q, %heads) : (tensor<1x2x512xf16>, tensor<3xi64>) -> tensor<1x8x128xf16>
  %qn = "onnx.Custom"(%qh, %qw) {function_name = "SimplifiedLayerNormalization", domain_name = "", epsilon = 9.99999997E-7 : f32, axis = -1 : si64, stash_type = 1 : si64} : (tensor<1x8x128xf16>, tensor<128xf16>) -> tensor<1x8x128xf16>
  %qr = "onnx.Reshape"(%qn, %qshape) : (tensor<1x8x128xf16>, tensor<3xi64>) -> tensor<1x2x512xf16>
  %kh = "onnx.Reshape"(%k, %heads) : (tensor<1x2x128xf16>, tensor<3xi64>) -> tensor<1x2x128xf16>
  %kn = "onnx.Custom"(%kh, %kw) {function_name = "SimplifiedLayerNormalization", domain_name = "", epsilon = 9.99999997E-7 : f32, axis = -1 : si64, stash_type = 1 : si64} : (tensor<1x2x128xf16>, tensor<128xf16>) -> tensor<1x2x128xf16>
  %kr = "onnx.Reshape"(%kn, %kshape) : (tensor<1x2x128xf16>, tensor<3xi64>) -> tensor<1x2x128xf16>
  %qrot = "onnx.Custom"(%qr, %pos, %cos, %sin) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", interleaved = 0 : si64, rotary_embedding_dim = 32 : si64, num_heads = 4 : si64, mrope_section = [6, 5, 5], mrope_layout = 1 : si64, is_packed_batching = 0 : si64} : (tensor<1x2x512xf16>, tensor<3x1x2xi64>, tensor<64x16xf16>, tensor<64x16xf16>) -> tensor<1x2x512xf16>
  %krot = "onnx.Custom"(%kr, %pos, %cos, %sin) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", interleaved = 0 : si64, rotary_embedding_dim = 32 : si64, num_heads = 1 : si64, mrope_section = [6, 5, 5], mrope_layout = 1 : si64, is_packed_batching = 0 : si64} : (tensor<1x2x128xf16>, tensor<3x1x2xi64>, tensor<64x16xf16>, tensor<64x16xf16>) -> tensor<1x2x128xf16>
  %out, %pk, %pv = "onnx.Custom"(%qrot, %krot, %v, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total, %none, %none, %none, %none, %none, %none) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32,
    attention_mode = "selected_only", selected_kv_source = "main"
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, none)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  %sig = "onnx.Sigmoid"(%g) : (tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
  %gated = "onnx.Mul"(%sig, %out) : (tensor<1x2x512xf16>, tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
  return %gated, %pk, %pv : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
}

// A gate computed after the attention moves the folded op to the Mul.
// CHECK-LABEL: func.func @late_gate(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[Q:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[K:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[V:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[G0:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[G1:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[IDX:[^:]*]]: tensor<2x8xi32>,
// CHECK-SAME:      %[[CNT:[^:]*]]: tensor<2xi32>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
// CHECK-NEXT:      %[[GATE_INIT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[GATE:.*]] = hip.add(%[[CTX]]) ins(%[[G0]], %[[G1]] : tensor<1x2x512xf16>, tensor<1x2x512xf16>) outs(%[[GATE_INIT]] : tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[R:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[Q]] : tensor<1x2x512xf16>) kv(%[[K]], %[[V]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[IDX]], %[[CNT]], %[[SEQLENS]] : tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          gate(%[[GATE]] : tensor<1x2x512xf16>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#1, %[[R]]#2 : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:    }
func.func @late_gate(
    %q: tensor<1x2x512xf16>, %k: tensor<1x2x128xf16>, %v: tensor<1x2x128xf16>,
    %g0: tensor<1x2x512xf16>, %g1: tensor<1x2x512xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %idx: tensor<2x8xi32>, %cnt: tensor<2xi32>, %seqlens: tensor<1xi32>,
    %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  %g = "onnx.Add"(%g0, %g1) : (tensor<1x2x512xf16>, tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
  %sig = "onnx.Sigmoid"(%g) : (tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
  %gated = "onnx.Mul"(%out, %sig) : (tensor<1x2x512xf16>, tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
  return %gated, %pk, %pv : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
}

// The present key is read before a late gate, so the gate stays unfolded.
// CHECK-LABEL: func.func @late_gate_after_present_read(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[Q:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[K:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[V:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[G0:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[IDX:[^:]*]]: tensor<2x8xi32>,
// CHECK-SAME:      %[[CNT:[^:]*]]: tensor<2xi32>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[R:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[Q]] : tensor<1x2x512xf16>) kv(%[[K]], %[[V]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[IDX]], %[[CNT]], %[[SEQLENS]] : tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[NEG_INIT:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[NEG:.*]] = hip.neg(%[[CTX]]) ins(%[[R]]#1 : tensor<1x1x16x128xf16>) outs(%[[NEG_INIT]] : tensor<1x1x16x128xf16>) : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[GATE_INIT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[GATE:.*]] = hip.add(%[[CTX]]) ins(%[[G0]], %[[G0]] : tensor<1x2x512xf16>, tensor<1x2x512xf16>) outs(%[[GATE_INIT]] : tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
// CHECK-NEXT:      %[[SIG_INIT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[SIG:.*]] = hip.sigmoid(%[[CTX]]) ins(%[[GATE]] : tensor<1x2x512xf16>) outs(%[[SIG_INIT]] : tensor<1x2x512xf16>) : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[MUL_INIT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[GATED:.*]] = hip.mul(%[[CTX]]) ins(%[[R]]#0, %[[SIG]] : tensor<1x2x512xf16>, tensor<1x2x512xf16>) outs(%[[MUL_INIT]] : tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
// CHECK-NEXT:      return %[[GATED]], %[[NEG]], %[[R]]#2 : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:    }
func.func @late_gate_after_present_read(
    %q: tensor<1x2x512xf16>, %k: tensor<1x2x128xf16>, %v: tensor<1x2x128xf16>,
    %g0: tensor<1x2x512xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %idx: tensor<2x8xi32>, %cnt: tensor<2xi32>, %seqlens: tensor<1xi32>,
    %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  %pk2 = "onnx.Neg"(%pk) : (tensor<1x1x16x128xf16>) -> tensor<1x1x16x128xf16>
  %g = "onnx.Add"(%g0, %g0) : (tensor<1x2x512xf16>, tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
  %sig = "onnx.Sigmoid"(%g) : (tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
  %gated = "onnx.Mul"(%out, %sig) : (tensor<1x2x512xf16>, tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
  return %gated, %pk2, %pv : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
}

// The attention output has a second user, so the gate stays unfolded.
// CHECK-LABEL: func.func @gate_output_reused(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[Q:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[K:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[V:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[G:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[IDX:[^:]*]]: tensor<2x8xi32>,
// CHECK-SAME:      %[[CNT:[^:]*]]: tensor<2xi32>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[R:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[Q]] : tensor<1x2x512xf16>) kv(%[[K]], %[[V]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[IDX]], %[[CNT]], %[[SEQLENS]] : tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[SIG_INIT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[SIG:.*]] = hip.sigmoid(%[[CTX]]) ins(%[[G]] : tensor<1x2x512xf16>) outs(%[[SIG_INIT]] : tensor<1x2x512xf16>) : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[MUL_INIT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[GATED:.*]] = hip.mul(%[[CTX]]) ins(%[[R]]#0, %[[SIG]] : tensor<1x2x512xf16>, tensor<1x2x512xf16>) outs(%[[MUL_INIT]] : tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
// CHECK-NEXT:      return %[[GATED]], %[[R]]#0, %[[R]]#1, %[[R]]#2 : tensor<1x2x512xf16>, tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:    }
func.func @gate_output_reused(
    %q: tensor<1x2x512xf16>, %k: tensor<1x2x128xf16>, %v: tensor<1x2x128xf16>,
    %g: tensor<1x2x512xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %idx: tensor<2x8xi32>, %cnt: tensor<2xi32>, %seqlens: tensor<1xi32>,
    %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  %sig = "onnx.Sigmoid"(%g) : (tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
  %gated = "onnx.Mul"(%out, %sig) : (tensor<1x2x512xf16>, tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
  return %gated, %out, %pk, %pv : tensor<1x2x512xf16>, tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
}

// The Split outputs become a packed query; a broadcasting gate stays unfolded.
// CHECK-LABEL: func.func @packed_broadcast_gate(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QKV:[^:]*]]: tensor<1x2x768xf16>,
// CHECK-SAME:      %[[G:[^:]*]]: tensor<1x1x512xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[IDX:[^:]*]]: tensor<2x8xi32>,
// CHECK-SAME:      %[[CNT:[^:]*]]: tensor<2xi32>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[R:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[QKV]] : tensor<1x2x768xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[IDX]], %[[CNT]], %[[SEQLENS]] : tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[SIG_INIT:.*]] = tensor.empty() : tensor<1x1x512xf16>
// CHECK-NEXT:      %[[SIG:.*]] = hip.sigmoid(%[[CTX]]) ins(%[[G]] : tensor<1x1x512xf16>) outs(%[[SIG_INIT]] : tensor<1x1x512xf16>) : tensor<1x1x512xf16>
// CHECK-NEXT:      %[[MUL_INIT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[GATED:.*]] = hip.mul(%[[CTX]]) ins(%[[R]]#0, %[[SIG]] : tensor<1x2x512xf16>, tensor<1x1x512xf16>) outs(%[[MUL_INIT]] : tensor<1x2x512xf16>) -> tensor<1x2x512xf16>
// CHECK-NEXT:      return %[[GATED]], %[[R]]#1, %[[R]]#2 : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:    }
func.func @packed_broadcast_gate(
    %qkv: tensor<1x2x768xf16>, %g: tensor<1x1x512xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %idx: tensor<2x8xi32>, %cnt: tensor<2xi32>, %seqlens: tensor<1xi32>,
    %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = "onnx.Constant"() {value = dense<[512, 128, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
  %q, %k, %v = "onnx.Split"(%qkv, %sizes) {axis = 2 : si64} : (tensor<1x2x768xf16>, tensor<3xi64>) -> (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>)
  %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  %sig = "onnx.Sigmoid"(%g) : (tensor<1x1x512xf16>) -> tensor<1x1x512xf16>
  %gated = "onnx.Mul"(%out, %sig) : (tensor<1x2x512xf16>, tensor<1x1x512xf16>) -> tensor<1x2x512xf16>
  return %gated, %pk, %pv : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
}

// k and v come from the Split out of order, so the query stays unpacked.
// CHECK-LABEL: func.func @split_order_swapped(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QKV:[^:]*]]: tensor<1x2x768xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[IDX:[^:]*]]: tensor<2x8xi32>,
// CHECK-SAME:      %[[CNT:[^:]*]]: tensor<2xi32>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
// CHECK-NEXT:      %[[Q:.*]] = tensor.extract_slice %[[QKV]][0, 0, 0] [1, 2, 512] [1, 1, 1] : tensor<1x2x768xf16> to tensor<1x2x512xf16>
// CHECK-NEXT:      %[[K:.*]] = tensor.extract_slice %[[QKV]][0, 0, 512] [1, 2, 128] [1, 1, 1] : tensor<1x2x768xf16> to tensor<1x2x128xf16>
// CHECK-NEXT:      %[[V:.*]] = tensor.extract_slice %[[QKV]][0, 0, 640] [1, 2, 128] [1, 1, 1] : tensor<1x2x768xf16> to tensor<1x2x128xf16>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[R:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[Q]] : tensor<1x2x512xf16>) kv(%[[V]], %[[K]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[IDX]], %[[CNT]], %[[SEQLENS]] : tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#1, %[[R]]#2 : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:    }
func.func @split_order_swapped(
    %qkv: tensor<1x2x768xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %idx: tensor<2x8xi32>, %cnt: tensor<2xi32>, %seqlens: tensor<1xi32>,
    %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = "onnx.Constant"() {value = dense<[512, 128, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
  %q, %k, %v = "onnx.Split"(%qkv, %sizes) {axis = -1 : si64} : (tensor<1x2x768xf16>, tensor<3xi64>) -> (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>)
  %out, %pk, %pv = "onnx.Custom"(%q, %v, %k, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  return %out, %pk, %pv : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
}

// q and k norms use different epsilons, so both chains stay unfolded.
// CHECK-LABEL: func.func @epsilon_mismatch(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[Q:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[K:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[V:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[POS:[^:]*]]: tensor<3x1x2xi64>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<64x16xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<64x16xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[IDX:[^:]*]]: tensor<2x8xi32>,
// CHECK-SAME:      %[[CNT:[^:]*]]: tensor<2xi32>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
// CHECK-NEXT:      %[[QW:.*]] = hip.constant {serialization_order = 11 : i64, value = dense<1.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[KW:.*]] = hip.constant {serialization_order = 12 : i64, value = dense<2.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[Q_FLAT:.*]] = tensor.collapse_shape %[[Q]] {{\[\[}}0, 1, 2]] : tensor<1x2x512xf16> into tensor<1024xf16>
// CHECK-NEXT:      %[[Q_HEADS:.*]] = tensor.expand_shape %[[Q_FLAT]] {{\[\[}}0, 1, 2]] output_shape [1, 8, 128] : tensor<1024xf16> into tensor<1x8x128xf16>
// CHECK-NEXT:      %[[QN_INIT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[QN:.*]] = hip.rms_norm(%[[CTX]]) ins(%[[Q_HEADS]], %[[QW]] : tensor<1x8x128xf16>, tensor<128xf16>) outs(%[[QN_INIT]] : tensor<1x8x128xf16>) {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64} : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[QN_FLAT:.*]] = tensor.collapse_shape %[[QN]] {{\[\[}}0, 1, 2]] : tensor<1x8x128xf16> into tensor<1024xf16>
// CHECK-NEXT:      %[[QN_MERGED:.*]] = tensor.expand_shape %[[QN_FLAT]] {{\[\[}}0, 1, 2]] output_shape [1, 2, 512] : tensor<1024xf16> into tensor<1x2x512xf16>
// CHECK-NEXT:      %[[KN_INIT:.*]] = tensor.empty() : tensor<1x2x128xf16>
// CHECK-NEXT:      %[[KN:.*]] = hip.rms_norm(%[[CTX]]) ins(%[[K]], %[[KW]] : tensor<1x2x128xf16>, tensor<128xf16>) outs(%[[KN_INIT]] : tensor<1x2x128xf16>) {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64} : tensor<1x2x128xf16>
// CHECK-NEXT:      %[[QROT_INIT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[QROT:.*]] = hip.mrotary_embedding(%[[CTX]]) ins(%[[QN_MERGED]], %[[POS]], %[[COS]], %[[SIN]] : tensor<1x2x512xf16>, tensor<3x1x2xi64>, tensor<64x16xf16>, tensor<64x16xf16>) outs(%[[QROT_INIT]] : tensor<1x2x512xf16>) {mrope_layout = 1 : i64, mrope_section = [6, 5, 5], num_heads = 4 : i64, rotary_embedding_dim = 32 : i64} : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[KROT_INIT:.*]] = tensor.empty() : tensor<1x2x128xf16>
// CHECK-NEXT:      %[[KROT:.*]] = hip.mrotary_embedding(%[[CTX]]) ins(%[[KN]], %[[POS]], %[[COS]], %[[SIN]] : tensor<1x2x128xf16>, tensor<3x1x2xi64>, tensor<64x16xf16>, tensor<64x16xf16>) outs(%[[KROT_INIT]] : tensor<1x2x128xf16>) {mrope_layout = 1 : i64, mrope_section = [6, 5, 5], num_heads = 1 : i64, rotary_embedding_dim = 32 : i64} : tensor<1x2x128xf16>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[R:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[QROT]] : tensor<1x2x512xf16>) kv(%[[KROT]], %[[V]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[IDX]], %[[CNT]], %[[SEQLENS]] : tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#1, %[[R]]#2 : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:    }
func.func @epsilon_mismatch(
    %q: tensor<1x2x512xf16>, %k: tensor<1x2x128xf16>, %v: tensor<1x2x128xf16>,
    %pos: tensor<3x1x2xi64>, %cos: tensor<64x16xf16>, %sin: tensor<64x16xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %idx: tensor<2x8xi32>, %cnt: tensor<2xi32>, %seqlens: tensor<1xi32>,
    %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %heads = "onnx.Constant"() {value = dense<[0, -1, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
  %qshape = "onnx.Constant"() {value = dense<[0, -1, 512]> : tensor<3xi64>} : () -> tensor<3xi64>
  %kshape = "onnx.Constant"() {value = dense<[0, -1, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
  %qw = "onnx.Constant"() {value = dense<1.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %kw = "onnx.Constant"() {value = dense<2.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %qh = "onnx.Reshape"(%q, %heads) : (tensor<1x2x512xf16>, tensor<3xi64>) -> tensor<1x8x128xf16>
  %qn = "onnx.Custom"(%qh, %qw) {function_name = "SimplifiedLayerNormalization", domain_name = "", epsilon = 9.99999997E-7 : f32, axis = -1 : si64, stash_type = 1 : si64} : (tensor<1x8x128xf16>, tensor<128xf16>) -> tensor<1x8x128xf16>
  %qr = "onnx.Reshape"(%qn, %qshape) : (tensor<1x8x128xf16>, tensor<3xi64>) -> tensor<1x2x512xf16>
  %kh = "onnx.Reshape"(%k, %heads) : (tensor<1x2x128xf16>, tensor<3xi64>) -> tensor<1x2x128xf16>
  %kn = "onnx.Custom"(%kh, %kw) {function_name = "SimplifiedLayerNormalization", domain_name = "", epsilon = 9.99999974E-6 : f32, axis = -1 : si64, stash_type = 1 : si64} : (tensor<1x2x128xf16>, tensor<128xf16>) -> tensor<1x2x128xf16>
  %kr = "onnx.Reshape"(%kn, %kshape) : (tensor<1x2x128xf16>, tensor<3xi64>) -> tensor<1x2x128xf16>
  %qrot = "onnx.Custom"(%qr, %pos, %cos, %sin) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", interleaved = 0 : si64, rotary_embedding_dim = 32 : si64, num_heads = 4 : si64, mrope_section = [6, 5, 5], mrope_layout = 1 : si64, is_packed_batching = 0 : si64} : (tensor<1x2x512xf16>, tensor<3x1x2xi64>, tensor<64x16xf16>, tensor<64x16xf16>) -> tensor<1x2x512xf16>
  %krot = "onnx.Custom"(%kr, %pos, %cos, %sin) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", interleaved = 0 : si64, rotary_embedding_dim = 32 : si64, num_heads = 1 : si64, mrope_section = [6, 5, 5], mrope_layout = 1 : si64, is_packed_batching = 0 : si64} : (tensor<1x2x128xf16>, tensor<3x1x2xi64>, tensor<64x16xf16>, tensor<64x16xf16>) -> tensor<1x2x128xf16>
  %out, %pk, %pv = "onnx.Custom"(%qrot, %krot, %v, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  return %out, %pk, %pv : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
}

// The k norm has a second user, so both chains stay unfolded.
// CHECK-LABEL: func.func @norm_output_reused(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[Q:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[K:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[V:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[POS:[^:]*]]: tensor<3x1x2xi64>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<64x16xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<64x16xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[IDX:[^:]*]]: tensor<2x8xi32>,
// CHECK-SAME:      %[[CNT:[^:]*]]: tensor<2xi32>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
// CHECK-NEXT:      %[[QW:.*]] = hip.constant {serialization_order = 16 : i64, value = dense<1.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[KW:.*]] = hip.constant {serialization_order = 17 : i64, value = dense<2.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[Q_FLAT:.*]] = tensor.collapse_shape %[[Q]] {{\[\[}}0, 1, 2]] : tensor<1x2x512xf16> into tensor<1024xf16>
// CHECK-NEXT:      %[[Q_HEADS:.*]] = tensor.expand_shape %[[Q_FLAT]] {{\[\[}}0, 1, 2]] output_shape [1, 8, 128] : tensor<1024xf16> into tensor<1x8x128xf16>
// CHECK-NEXT:      %[[QN_INIT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[QN:.*]] = hip.rms_norm(%[[CTX]]) ins(%[[Q_HEADS]], %[[QW]] : tensor<1x8x128xf16>, tensor<128xf16>) outs(%[[QN_INIT]] : tensor<1x8x128xf16>) {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64} : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[QN_FLAT:.*]] = tensor.collapse_shape %[[QN]] {{\[\[}}0, 1, 2]] : tensor<1x8x128xf16> into tensor<1024xf16>
// CHECK-NEXT:      %[[QN_MERGED:.*]] = tensor.expand_shape %[[QN_FLAT]] {{\[\[}}0, 1, 2]] output_shape [1, 2, 512] : tensor<1024xf16> into tensor<1x2x512xf16>
// CHECK-NEXT:      %[[KN_INIT:.*]] = tensor.empty() : tensor<1x2x128xf16>
// CHECK-NEXT:      %[[KN:.*]] = hip.rms_norm(%[[CTX]]) ins(%[[K]], %[[KW]] : tensor<1x2x128xf16>, tensor<128xf16>) outs(%[[KN_INIT]] : tensor<1x2x128xf16>) {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64} : tensor<1x2x128xf16>
// CHECK-NEXT:      %[[QROT_INIT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[QROT:.*]] = hip.mrotary_embedding(%[[CTX]]) ins(%[[QN_MERGED]], %[[POS]], %[[COS]], %[[SIN]] : tensor<1x2x512xf16>, tensor<3x1x2xi64>, tensor<64x16xf16>, tensor<64x16xf16>) outs(%[[QROT_INIT]] : tensor<1x2x512xf16>) {mrope_layout = 1 : i64, mrope_section = [6, 5, 5], num_heads = 4 : i64, rotary_embedding_dim = 32 : i64} : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[KROT_INIT:.*]] = tensor.empty() : tensor<1x2x128xf16>
// CHECK-NEXT:      %[[KROT:.*]] = hip.mrotary_embedding(%[[CTX]]) ins(%[[KN]], %[[POS]], %[[COS]], %[[SIN]] : tensor<1x2x128xf16>, tensor<3x1x2xi64>, tensor<64x16xf16>, tensor<64x16xf16>) outs(%[[KROT_INIT]] : tensor<1x2x128xf16>) {mrope_layout = 1 : i64, mrope_section = [6, 5, 5], num_heads = 1 : i64, rotary_embedding_dim = 32 : i64} : tensor<1x2x128xf16>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[R:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[QROT]] : tensor<1x2x512xf16>) kv(%[[KROT]], %[[V]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[IDX]], %[[CNT]], %[[SEQLENS]] : tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      return %[[R]]#0, %[[KN]], %[[R]]#1, %[[R]]#2 : tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:    }
func.func @norm_output_reused(
    %q: tensor<1x2x512xf16>, %k: tensor<1x2x128xf16>, %v: tensor<1x2x128xf16>,
    %pos: tensor<3x1x2xi64>, %cos: tensor<64x16xf16>, %sin: tensor<64x16xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %idx: tensor<2x8xi32>, %cnt: tensor<2xi32>, %seqlens: tensor<1xi32>,
    %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %heads = "onnx.Constant"() {value = dense<[0, -1, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
  %qshape = "onnx.Constant"() {value = dense<[0, -1, 512]> : tensor<3xi64>} : () -> tensor<3xi64>
  %kshape = "onnx.Constant"() {value = dense<[0, -1, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
  %qw = "onnx.Constant"() {value = dense<1.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %kw = "onnx.Constant"() {value = dense<2.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %qh = "onnx.Reshape"(%q, %heads) : (tensor<1x2x512xf16>, tensor<3xi64>) -> tensor<1x8x128xf16>
  %qn = "onnx.Custom"(%qh, %qw) {function_name = "SimplifiedLayerNormalization", domain_name = "", epsilon = 9.99999997E-7 : f32, axis = -1 : si64, stash_type = 1 : si64} : (tensor<1x8x128xf16>, tensor<128xf16>) -> tensor<1x8x128xf16>
  %qr = "onnx.Reshape"(%qn, %qshape) : (tensor<1x8x128xf16>, tensor<3xi64>) -> tensor<1x2x512xf16>
  %kh = "onnx.Reshape"(%k, %heads) : (tensor<1x2x128xf16>, tensor<3xi64>) -> tensor<1x2x128xf16>
  %kn = "onnx.Custom"(%kh, %kw) {function_name = "SimplifiedLayerNormalization", domain_name = "", epsilon = 9.99999997E-7 : f32, axis = -1 : si64, stash_type = 1 : si64} : (tensor<1x2x128xf16>, tensor<128xf16>) -> tensor<1x2x128xf16>
  %kr = "onnx.Reshape"(%kn, %kshape) : (tensor<1x2x128xf16>, tensor<3xi64>) -> tensor<1x2x128xf16>
  %qrot = "onnx.Custom"(%qr, %pos, %cos, %sin) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", interleaved = 0 : si64, rotary_embedding_dim = 32 : si64, num_heads = 4 : si64, mrope_section = [6, 5, 5], mrope_layout = 1 : si64, is_packed_batching = 0 : si64} : (tensor<1x2x512xf16>, tensor<3x1x2xi64>, tensor<64x16xf16>, tensor<64x16xf16>) -> tensor<1x2x512xf16>
  %krot = "onnx.Custom"(%kr, %pos, %cos, %sin) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", interleaved = 0 : si64, rotary_embedding_dim = 32 : si64, num_heads = 1 : si64, mrope_section = [6, 5, 5], mrope_layout = 1 : si64, is_packed_batching = 0 : si64} : (tensor<1x2x128xf16>, tensor<3x1x2xi64>, tensor<64x16xf16>, tensor<64x16xf16>) -> tensor<1x2x128xf16>
  %out, %pk, %pv = "onnx.Custom"(%qrot, %krot, %v, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x8xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  return %out, %kn, %pk, %pv : tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
}
