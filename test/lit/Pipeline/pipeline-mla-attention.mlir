// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify an ONNX graph that expands K/V from a latent cache [B, 1, T, L + R]
// (Squeeze -> Split [L, R] -> up-projections, rope key broadcast over heads)
// into an Attention that is causal over the first nonpad_kv_seqlen rows ends
// up as hip.mla_attention on the cache: convert-onnx-to-hip lowers the
// Attention to hip.kv_attention, and hip-fusion-transform folds the expansion
// into it, for fp16 MatMul and int4 MatMulNBits up-projections and for a
// static or dynamic cache length. The cache is a TensorScatter-written buffer.
// Each negative case breaks one fact the fusion depends on and keeps the
// expansion:
//   - the Attention takes an explicit mask instead of nonpad_kv_seqlen
//     (hip.gqa),
//   - nonpad_kv_seqlen without is_causal (no HIP lowering),
//   - V is not projected from the cached latent,
//   - a MatMulNBits up-projection has zero points,
//   - an up-projection weight is not a constant,
//   - the Attention head split differs from the expansion's
//     (all four: hip.kv_attention).
// Geometry: H = 2, L = 32, R = 8, dn = 8, dv = 16, 64 cache rows.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --hip-infer-shapes --hip-fusion-transform --split-input-file | FileCheck %s

// ===== fp16 MatMul up-projections =====
module {
  func.func @main_graph(%q: tensor<1x?x32xf16>, %past: tensor<1x1x64x40xf16>,
                        %new: tensor<1x1x?x40xf16>, %m: tensor<1x?xi64>)
      -> (tensor<1x?x32xf16>, tensor<1x1x64x40xf16>) {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[Q:[^:]*]]: tensor<1x?x32xf16>, %[[PAST:[^:]*]]: tensor<1x1x64x40xf16>, %[[NEW:[^:]*]]: tensor<1x1x?x40xf16>
    // CHECK: %[[CACHE:.*]] = hip.tensor_scatter(%[[CTX]]) ins(%[[PAST]], %[[NEW]] : tensor<1x1x64x40xf16>, tensor<1x1x?x40xf16>) write_indices(%{{.*}} : tensor<1xi64>)
    // CHECK-SAME: {axis = 2 : i64}
    // CHECK: hip.mla_attention(%[[CTX]]) ins(%[[Q]], %[[CACHE]], %{{.*}}, %{{.*}} : tensor<1x?x32xf16>, tensor<1x1x64x40xf16>, tensor<32x16xf16>, tensor<32x32xf16>)
    // CHECK-SAME: kv_seqlen(%{{.*}} : tensor<1xi64>)
    // CHECK-SAME: {kv_lora_rank = 32 : i64, num_heads = 2 : i64, qk_nope_dim = 8 : i64, qk_rope_dim = 8 : i64, scale = 2.500000e-01 : f32, v_dim = 16 : i64
    // CHECK-NOT: hip.kv_attention
    // CHECK-NOT: hip.matmul
    // CHECK-NOT: onnx.
    // CHECK: return %{{.*}}, %[[CACHE]]
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %nonpad = "onnx.ReduceSum"(%m, %c1) {keepdims = 0 : si64} : (tensor<1x?xi64>, tensor<1xi64>) -> tensor<1xi64>
    %slen = "onnx.Shape"(%new) {end = 3 : si64, start = 2 : si64} : (tensor<1x1x?x40xf16>) -> tensor<1xi64>
    %wi = "onnx.Sub"(%nonpad, %slen) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %cache = "onnx.TensorScatter"(%past, %new, %wi) {axis = 2 : si64}
        : (tensor<1x1x64x40xf16>, tensor<1x1x?x40xf16>, tensor<1xi64>) -> tensor<1x1x64x40xf16>

    %split = "onnx.Constant"() {value = dense<[32, 8]> : tensor<2xi64>} : () -> tensor<2xi64>
    %shkn = "onnx.Constant"() {value = dense<[0, 0, 2, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %shkr = "onnx.Constant"() {value = dense<[0, 0, 1, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %heads = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %flat = "onnx.Constant"() {value = dense<[0, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %wk = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<32x16xf16>} : () -> tensor<32x16xf16>
    %wv = "onnx.Constant"() {value = dense<2.000000e-02> : tensor<32x32xf16>} : () -> tensor<32x32xf16>
    %lat = "onnx.Squeeze"(%cache, %c1) : (tensor<1x1x64x40xf16>, tensor<1xi64>) -> tensor<1x64x40xf16>
    %c, %kr = "onnx.Split"(%lat, %split) {axis = -1 : si64} : (tensor<1x64x40xf16>, tensor<2xi64>) -> (tensor<1x64x32xf16>, tensor<1x64x8xf16>)
    %kn = "onnx.MatMul"(%c, %wk) : (tensor<1x64x32xf16>, tensor<32x16xf16>) -> tensor<1x64x16xf16>
    %kn4 = "onnx.Reshape"(%kn, %shkn) : (tensor<1x64x16xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %kr4 = "onnx.Reshape"(%kr, %shkr) : (tensor<1x64x8xf16>, tensor<4xi64>) -> tensor<1x64x1x8xf16>
    %krh = "onnx.Expand"(%kr4, %heads) : (tensor<1x64x1x8xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %k4 = "onnx.Concat"(%kn4, %krh) {axis = -1 : si64} : (tensor<1x64x2x8xf16>, tensor<1x64x2x8xf16>) -> tensor<1x64x2x16xf16>
    %k = "onnx.Reshape"(%k4, %flat) : (tensor<1x64x2x16xf16>, tensor<3xi64>) -> tensor<1x64x32xf16>
    %v = "onnx.MatMul"(%c, %wv) : (tensor<1x64x32xf16>, tensor<32x32xf16>) -> tensor<1x64x32xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %nonpad)
        {is_causal = 1 : si64, kv_num_heads = 2 : si64, q_num_heads = 2 : si64, scale = 2.500000e-01 : f32}
        : (tensor<1x?x32xf16>, tensor<1x64x32xf16>, tensor<1x64x32xf16>, none, none, none, tensor<1xi64>)
        -> tensor<1x?x32xf16>
    return %y, %cache : tensor<1x?x32xf16>, tensor<1x1x64x40xf16>
  }
}

// -----

// ===== int4 MatMulNBits up-projections (block 16) =====
module {
  func.func @main_graph(%q: tensor<1x?x32xf16>, %past: tensor<1x1x64x40xf16>,
                        %new: tensor<1x1x?x40xf16>, %m: tensor<1x?xi64>)
      -> (tensor<1x?x32xf16>, tensor<1x1x64x40xf16>) {
    // CHECK-LABEL: func.func @main_graph
    // CHECK: hip.mla_attention(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x?x32xf16>, tensor<1x1x64x40xf16>, tensor<16x2x8xui8>, tensor<32x2x8xui8>)
    // CHECK-SAME: k_b_scales(%{{.*}} : tensor<16x2xf16>) v_b_scales(%{{.*}} : tensor<32x2xf16>) kv_seqlen(%{{.*}} : tensor<1xi64>)
    // CHECK-SAME: block_size = 16 : i64
    // CHECK-NOT: hip.kv_attention
    // CHECK-NOT: hip.matmul_nbits
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %nonpad = "onnx.ReduceSum"(%m, %c1) {keepdims = 0 : si64} : (tensor<1x?xi64>, tensor<1xi64>) -> tensor<1xi64>
    %slen = "onnx.Shape"(%new) {end = 3 : si64, start = 2 : si64} : (tensor<1x1x?x40xf16>) -> tensor<1xi64>
    %wi = "onnx.Sub"(%nonpad, %slen) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %cache = "onnx.TensorScatter"(%past, %new, %wi) {axis = 2 : si64}
        : (tensor<1x1x64x40xf16>, tensor<1x1x?x40xf16>, tensor<1xi64>) -> tensor<1x1x64x40xf16>

    %split = "onnx.Constant"() {value = dense<[32, 8]> : tensor<2xi64>} : () -> tensor<2xi64>
    %shkn = "onnx.Constant"() {value = dense<[0, 0, 2, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %shkr = "onnx.Constant"() {value = dense<[0, 0, 1, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %heads = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %flat = "onnx.Constant"() {value = dense<[0, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %wk = "onnx.Constant"() {value = dense<136> : tensor<16x2x8xui8>} : () -> tensor<16x2x8xui8>
    %sk = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<16x2xf16>} : () -> tensor<16x2xf16>
    %wv = "onnx.Constant"() {value = dense<137> : tensor<32x2x8xui8>} : () -> tensor<32x2x8xui8>
    %sv = "onnx.Constant"() {value = dense<2.000000e-02> : tensor<32x2xf16>} : () -> tensor<32x2xf16>
    %lat = "onnx.Squeeze"(%cache, %c1) : (tensor<1x1x64x40xf16>, tensor<1xi64>) -> tensor<1x64x40xf16>
    %c, %kr = "onnx.Split"(%lat, %split) {axis = -1 : si64} : (tensor<1x64x40xf16>, tensor<2xi64>) -> (tensor<1x64x32xf16>, tensor<1x64x8xf16>)
    %kn = "onnx.Custom"(%c, %wk, %sk) {function_name = "MatMulNBits", domain_name = "com.microsoft", K = 32 : si64, N = 16 : si64, bits = 4 : si64, block_size = 16 : si64} : (tensor<1x64x32xf16>, tensor<16x2x8xui8>, tensor<16x2xf16>) -> tensor<1x64x16xf16>
    %kn4 = "onnx.Reshape"(%kn, %shkn) : (tensor<1x64x16xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %kr4 = "onnx.Reshape"(%kr, %shkr) : (tensor<1x64x8xf16>, tensor<4xi64>) -> tensor<1x64x1x8xf16>
    %krh = "onnx.Expand"(%kr4, %heads) : (tensor<1x64x1x8xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %k4 = "onnx.Concat"(%kn4, %krh) {axis = 3 : si64} : (tensor<1x64x2x8xf16>, tensor<1x64x2x8xf16>) -> tensor<1x64x2x16xf16>
    %k = "onnx.Reshape"(%k4, %flat) : (tensor<1x64x2x16xf16>, tensor<3xi64>) -> tensor<1x64x32xf16>
    %v = "onnx.Custom"(%c, %wv, %sv) {function_name = "MatMulNBits", domain_name = "com.microsoft", K = 32 : si64, N = 32 : si64, bits = 4 : si64, block_size = 16 : si64} : (tensor<1x64x32xf16>, tensor<32x2x8xui8>, tensor<32x2xf16>) -> tensor<1x64x32xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %nonpad)
        {is_causal = 1 : si64, kv_num_heads = 2 : si64, q_num_heads = 2 : si64, scale = 2.500000e-01 : f32}
        : (tensor<1x?x32xf16>, tensor<1x64x32xf16>, tensor<1x64x32xf16>, none, none, none, tensor<1xi64>)
        -> tensor<1x?x32xf16>
    return %y, %cache : tensor<1x?x32xf16>, tensor<1x1x64x40xf16>
  }
}

// -----

// ===== Dynamic cache length =====
module {
  func.func @main_graph(%q: tensor<1x?x32xf16>, %past: tensor<1x1x?x40xf16>,
                        %new: tensor<1x1x?x40xf16>, %m: tensor<1x?xi64>)
      -> (tensor<1x?x32xf16>, tensor<1x1x?x40xf16>) {
    // CHECK-LABEL: func.func @main_graph
    // CHECK: %[[CACHE:.*]] = hip.tensor_scatter
    // CHECK: hip.mla_attention(%{{.*}}) ins(%{{.*}}, %[[CACHE]], %{{.*}}, %{{.*}} : tensor<1x?x32xf16>, tensor<1x1x?x40xf16>, tensor<32x16xf16>, tensor<32x32xf16>)
    // CHECK-SAME: kv_seqlen(%{{.*}} : tensor<1xi64>)
    // CHECK-NOT: hip.kv_attention
    // CHECK-NOT: hip.matmul
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %nonpad = "onnx.ReduceSum"(%m, %c1) {keepdims = 0 : si64} : (tensor<1x?xi64>, tensor<1xi64>) -> tensor<1xi64>
    %slen = "onnx.Shape"(%new) {end = 3 : si64, start = 2 : si64} : (tensor<1x1x?x40xf16>) -> tensor<1xi64>
    %wi = "onnx.Sub"(%nonpad, %slen) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %cache = "onnx.TensorScatter"(%past, %new, %wi) {axis = 2 : si64}
        : (tensor<1x1x?x40xf16>, tensor<1x1x?x40xf16>, tensor<1xi64>) -> tensor<1x1x?x40xf16>
    %split = "onnx.Constant"() {value = dense<[32, 8]> : tensor<2xi64>} : () -> tensor<2xi64>
    %shkn = "onnx.Constant"() {value = dense<[0, 0, 2, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %shkr = "onnx.Constant"() {value = dense<[0, 0, 1, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %heads = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %flat = "onnx.Constant"() {value = dense<[0, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %wk = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<32x16xf16>} : () -> tensor<32x16xf16>
    %wv = "onnx.Constant"() {value = dense<2.000000e-02> : tensor<32x32xf16>} : () -> tensor<32x32xf16>
    %lat = "onnx.Squeeze"(%cache, %c1) : (tensor<1x1x?x40xf16>, tensor<1xi64>) -> tensor<1x?x40xf16>
    %c, %kr = "onnx.Split"(%lat, %split) {axis = -1 : si64} : (tensor<1x?x40xf16>, tensor<2xi64>) -> (tensor<1x?x32xf16>, tensor<1x?x8xf16>)
    %kn = "onnx.MatMul"(%c, %wk) : (tensor<1x?x32xf16>, tensor<32x16xf16>) -> tensor<1x?x16xf16>
    %kn4 = "onnx.Reshape"(%kn, %shkn) : (tensor<1x?x16xf16>, tensor<4xi64>) -> tensor<1x?x2x8xf16>
    %kr4 = "onnx.Reshape"(%kr, %shkr) : (tensor<1x?x8xf16>, tensor<4xi64>) -> tensor<1x?x1x8xf16>
    %krh = "onnx.Expand"(%kr4, %heads) : (tensor<1x?x1x8xf16>, tensor<4xi64>) -> tensor<1x?x2x8xf16>
    %k4 = "onnx.Concat"(%kn4, %krh) {axis = -1 : si64} : (tensor<1x?x2x8xf16>, tensor<1x?x2x8xf16>) -> tensor<1x?x2x16xf16>
    %k = "onnx.Reshape"(%k4, %flat) : (tensor<1x?x2x16xf16>, tensor<3xi64>) -> tensor<1x?x32xf16>
    %v = "onnx.MatMul"(%c, %wv) : (tensor<1x?x32xf16>, tensor<32x32xf16>) -> tensor<1x?x32xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %nonpad)
        {is_causal = 1 : si64, kv_num_heads = 2 : si64, q_num_heads = 2 : si64, scale = 2.500000e-01 : f32}
        : (tensor<1x?x32xf16>, tensor<1x?x32xf16>, tensor<1x?x32xf16>, none, none, none, tensor<1xi64>)
        -> tensor<1x?x32xf16>
    return %y, %cache : tensor<1x?x32xf16>, tensor<1x1x?x40xf16>
  }
}

// -----

// ===== Negative: an explicit mask instead of nonpad_kv_seqlen; whether it
// masks any key is only known at run time =====
module {
  func.func @main_graph(%q: tensor<1x?x32xf16>, %cache: tensor<1x1x64x40xf16>,
                        %bias: tensor<1x1x?x64xf16>)
      -> tensor<1x?x32xf16> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: hip.mla_attention
    // CHECK: hip.gqa
    // CHECK-NOT: hip.mla_attention
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %split = "onnx.Constant"() {value = dense<[32, 8]> : tensor<2xi64>} : () -> tensor<2xi64>
    %shkn = "onnx.Constant"() {value = dense<[0, 0, 2, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %shkr = "onnx.Constant"() {value = dense<[0, 0, 1, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %heads = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %flat = "onnx.Constant"() {value = dense<[0, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %wk = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<32x16xf16>} : () -> tensor<32x16xf16>
    %wv = "onnx.Constant"() {value = dense<2.000000e-02> : tensor<32x32xf16>} : () -> tensor<32x32xf16>
    %lat = "onnx.Squeeze"(%cache, %c1) : (tensor<1x1x64x40xf16>, tensor<1xi64>) -> tensor<1x64x40xf16>
    %c, %kr = "onnx.Split"(%lat, %split) {axis = -1 : si64} : (tensor<1x64x40xf16>, tensor<2xi64>) -> (tensor<1x64x32xf16>, tensor<1x64x8xf16>)
    %kn = "onnx.MatMul"(%c, %wk) : (tensor<1x64x32xf16>, tensor<32x16xf16>) -> tensor<1x64x16xf16>
    %kn4 = "onnx.Reshape"(%kn, %shkn) : (tensor<1x64x16xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %kr4 = "onnx.Reshape"(%kr, %shkr) : (tensor<1x64x8xf16>, tensor<4xi64>) -> tensor<1x64x1x8xf16>
    %krh = "onnx.Expand"(%kr4, %heads) : (tensor<1x64x1x8xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %k4 = "onnx.Concat"(%kn4, %krh) {axis = -1 : si64} : (tensor<1x64x2x8xf16>, tensor<1x64x2x8xf16>) -> tensor<1x64x2x16xf16>
    %k = "onnx.Reshape"(%k4, %flat) : (tensor<1x64x2x16xf16>, tensor<3xi64>) -> tensor<1x64x32xf16>
    %v = "onnx.MatMul"(%c, %wv) : (tensor<1x64x32xf16>, tensor<32x32xf16>) -> tensor<1x64x32xf16>
    %y = "onnx.Attention"(%q, %k, %v, %bias)
        {kv_num_heads = 2 : si64, q_num_heads = 2 : si64, scale = 2.500000e-01 : f32}
        : (tensor<1x?x32xf16>, tensor<1x64x32xf16>, tensor<1x64x32xf16>, tensor<1x1x?x64xf16>)
        -> tensor<1x?x32xf16>
    return %y : tensor<1x?x32xf16>
  }
}

// -----

// ===== Negative: nonpad_kv_seqlen without is_causal =====
module {
  func.func @main_graph(%q: tensor<1x?x32xf16>, %cache: tensor<1x1x64x40xf16>,
                        %nonpad: tensor<1xi64>)
      -> tensor<1x?x32xf16> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: hip.mla_attention
    // CHECK: onnx.Attention
    // CHECK-NOT: hip.mla_attention
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %split = "onnx.Constant"() {value = dense<[32, 8]> : tensor<2xi64>} : () -> tensor<2xi64>
    %shkn = "onnx.Constant"() {value = dense<[0, 0, 2, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %shkr = "onnx.Constant"() {value = dense<[0, 0, 1, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %heads = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %flat = "onnx.Constant"() {value = dense<[0, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %wk = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<32x16xf16>} : () -> tensor<32x16xf16>
    %wv = "onnx.Constant"() {value = dense<2.000000e-02> : tensor<32x32xf16>} : () -> tensor<32x32xf16>
    %lat = "onnx.Squeeze"(%cache, %c1) : (tensor<1x1x64x40xf16>, tensor<1xi64>) -> tensor<1x64x40xf16>
    %c, %kr = "onnx.Split"(%lat, %split) {axis = -1 : si64} : (tensor<1x64x40xf16>, tensor<2xi64>) -> (tensor<1x64x32xf16>, tensor<1x64x8xf16>)
    %kn = "onnx.MatMul"(%c, %wk) : (tensor<1x64x32xf16>, tensor<32x16xf16>) -> tensor<1x64x16xf16>
    %kn4 = "onnx.Reshape"(%kn, %shkn) : (tensor<1x64x16xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %kr4 = "onnx.Reshape"(%kr, %shkr) : (tensor<1x64x8xf16>, tensor<4xi64>) -> tensor<1x64x1x8xf16>
    %krh = "onnx.Expand"(%kr4, %heads) : (tensor<1x64x1x8xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %k4 = "onnx.Concat"(%kn4, %krh) {axis = -1 : si64} : (tensor<1x64x2x8xf16>, tensor<1x64x2x8xf16>) -> tensor<1x64x2x16xf16>
    %k = "onnx.Reshape"(%k4, %flat) : (tensor<1x64x2x16xf16>, tensor<3xi64>) -> tensor<1x64x32xf16>
    %v = "onnx.MatMul"(%c, %wv) : (tensor<1x64x32xf16>, tensor<32x32xf16>) -> tensor<1x64x32xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %nonpad)
        {kv_num_heads = 2 : si64, q_num_heads = 2 : si64, scale = 2.500000e-01 : f32}
        : (tensor<1x?x32xf16>, tensor<1x64x32xf16>, tensor<1x64x32xf16>, none, none, none, tensor<1xi64>)
        -> tensor<1x?x32xf16>
    return %y : tensor<1x?x32xf16>
  }
}

// -----

// ===== Negative: V is not projected from the cached latent =====
module {
  func.func @main_graph(%q: tensor<1x?x32xf16>, %cache: tensor<1x1x64x40xf16>,
                        %nonpad: tensor<1xi64>, %v: tensor<1x64x32xf16>)
      -> tensor<1x?x32xf16> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: hip.mla_attention
    // CHECK: hip.kv_attention
    // CHECK-NOT: hip.mla_attention
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %split = "onnx.Constant"() {value = dense<[32, 8]> : tensor<2xi64>} : () -> tensor<2xi64>
    %shkn = "onnx.Constant"() {value = dense<[0, 0, 2, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %shkr = "onnx.Constant"() {value = dense<[0, 0, 1, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %heads = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %flat = "onnx.Constant"() {value = dense<[0, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %wk = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<32x16xf16>} : () -> tensor<32x16xf16>
    %lat = "onnx.Squeeze"(%cache, %c1) : (tensor<1x1x64x40xf16>, tensor<1xi64>) -> tensor<1x64x40xf16>
    %c, %kr = "onnx.Split"(%lat, %split) {axis = -1 : si64} : (tensor<1x64x40xf16>, tensor<2xi64>) -> (tensor<1x64x32xf16>, tensor<1x64x8xf16>)
    %kn = "onnx.MatMul"(%c, %wk) : (tensor<1x64x32xf16>, tensor<32x16xf16>) -> tensor<1x64x16xf16>
    %kn4 = "onnx.Reshape"(%kn, %shkn) : (tensor<1x64x16xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %kr4 = "onnx.Reshape"(%kr, %shkr) : (tensor<1x64x8xf16>, tensor<4xi64>) -> tensor<1x64x1x8xf16>
    %krh = "onnx.Expand"(%kr4, %heads) : (tensor<1x64x1x8xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %k4 = "onnx.Concat"(%kn4, %krh) {axis = -1 : si64} : (tensor<1x64x2x8xf16>, tensor<1x64x2x8xf16>) -> tensor<1x64x2x16xf16>
    %k = "onnx.Reshape"(%k4, %flat) : (tensor<1x64x2x16xf16>, tensor<3xi64>) -> tensor<1x64x32xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %nonpad)
        {is_causal = 1 : si64, kv_num_heads = 2 : si64, q_num_heads = 2 : si64, scale = 2.500000e-01 : f32}
        : (tensor<1x?x32xf16>, tensor<1x64x32xf16>, tensor<1x64x32xf16>, none, none, none, tensor<1xi64>)
        -> tensor<1x?x32xf16>
    return %y : tensor<1x?x32xf16>
  }
}

// -----

// ===== Negative: MatMulNBits up-projection with zero points =====
module {
  func.func @main_graph(%q: tensor<1x?x32xf16>, %cache: tensor<1x1x64x40xf16>,
                        %nonpad: tensor<1xi64>)
      -> tensor<1x?x32xf16> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: hip.mla_attention
    // CHECK: hip.kv_attention
    // CHECK-NOT: hip.mla_attention
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %split = "onnx.Constant"() {value = dense<[32, 8]> : tensor<2xi64>} : () -> tensor<2xi64>
    %shkn = "onnx.Constant"() {value = dense<[0, 0, 2, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %shkr = "onnx.Constant"() {value = dense<[0, 0, 1, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %heads = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %flat = "onnx.Constant"() {value = dense<[0, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %wk = "onnx.Constant"() {value = dense<136> : tensor<16x2x8xui8>} : () -> tensor<16x2x8xui8>
    %sk = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<16x2xf16>} : () -> tensor<16x2xf16>
    %zk = "onnx.Constant"() {value = dense<136> : tensor<16x1xui8>} : () -> tensor<16x1xui8>
    %wv = "onnx.Constant"() {value = dense<137> : tensor<32x2x8xui8>} : () -> tensor<32x2x8xui8>
    %sv = "onnx.Constant"() {value = dense<2.000000e-02> : tensor<32x2xf16>} : () -> tensor<32x2xf16>
    %lat = "onnx.Squeeze"(%cache, %c1) : (tensor<1x1x64x40xf16>, tensor<1xi64>) -> tensor<1x64x40xf16>
    %c, %kr = "onnx.Split"(%lat, %split) {axis = -1 : si64} : (tensor<1x64x40xf16>, tensor<2xi64>) -> (tensor<1x64x32xf16>, tensor<1x64x8xf16>)
    %kn = "onnx.Custom"(%c, %wk, %sk, %zk) {function_name = "MatMulNBits", domain_name = "com.microsoft", K = 32 : si64, N = 16 : si64, bits = 4 : si64, block_size = 16 : si64} : (tensor<1x64x32xf16>, tensor<16x2x8xui8>, tensor<16x2xf16>, tensor<16x1xui8>) -> tensor<1x64x16xf16>
    %kn4 = "onnx.Reshape"(%kn, %shkn) : (tensor<1x64x16xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %kr4 = "onnx.Reshape"(%kr, %shkr) : (tensor<1x64x8xf16>, tensor<4xi64>) -> tensor<1x64x1x8xf16>
    %krh = "onnx.Expand"(%kr4, %heads) : (tensor<1x64x1x8xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %k4 = "onnx.Concat"(%kn4, %krh) {axis = -1 : si64} : (tensor<1x64x2x8xf16>, tensor<1x64x2x8xf16>) -> tensor<1x64x2x16xf16>
    %k = "onnx.Reshape"(%k4, %flat) : (tensor<1x64x2x16xf16>, tensor<3xi64>) -> tensor<1x64x32xf16>
    %v = "onnx.Custom"(%c, %wv, %sv) {function_name = "MatMulNBits", domain_name = "com.microsoft", K = 32 : si64, N = 32 : si64, bits = 4 : si64, block_size = 16 : si64} : (tensor<1x64x32xf16>, tensor<32x2x8xui8>, tensor<32x2xf16>) -> tensor<1x64x32xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %nonpad)
        {is_causal = 1 : si64, kv_num_heads = 2 : si64, q_num_heads = 2 : si64, scale = 2.500000e-01 : f32}
        : (tensor<1x?x32xf16>, tensor<1x64x32xf16>, tensor<1x64x32xf16>, none, none, none, tensor<1xi64>)
        -> tensor<1x?x32xf16>
    return %y : tensor<1x?x32xf16>
  }
}

// -----

// ===== Negative: the key up-projection weight is a graph input =====
module {
  func.func @main_graph(%q: tensor<1x?x32xf16>, %cache: tensor<1x1x64x40xf16>,
                        %nonpad: tensor<1xi64>, %wk: tensor<32x16xf16>)
      -> tensor<1x?x32xf16> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: hip.mla_attention
    // CHECK: hip.kv_attention
    // CHECK-NOT: hip.mla_attention
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %split = "onnx.Constant"() {value = dense<[32, 8]> : tensor<2xi64>} : () -> tensor<2xi64>
    %shkn = "onnx.Constant"() {value = dense<[0, 0, 2, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %shkr = "onnx.Constant"() {value = dense<[0, 0, 1, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %heads = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %flat = "onnx.Constant"() {value = dense<[0, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %wv = "onnx.Constant"() {value = dense<2.000000e-02> : tensor<32x32xf16>} : () -> tensor<32x32xf16>
    %lat = "onnx.Squeeze"(%cache, %c1) : (tensor<1x1x64x40xf16>, tensor<1xi64>) -> tensor<1x64x40xf16>
    %c, %kr = "onnx.Split"(%lat, %split) {axis = -1 : si64} : (tensor<1x64x40xf16>, tensor<2xi64>) -> (tensor<1x64x32xf16>, tensor<1x64x8xf16>)
    %kn = "onnx.MatMul"(%c, %wk) : (tensor<1x64x32xf16>, tensor<32x16xf16>) -> tensor<1x64x16xf16>
    %kn4 = "onnx.Reshape"(%kn, %shkn) : (tensor<1x64x16xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %kr4 = "onnx.Reshape"(%kr, %shkr) : (tensor<1x64x8xf16>, tensor<4xi64>) -> tensor<1x64x1x8xf16>
    %krh = "onnx.Expand"(%kr4, %heads) : (tensor<1x64x1x8xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %k4 = "onnx.Concat"(%kn4, %krh) {axis = -1 : si64} : (tensor<1x64x2x8xf16>, tensor<1x64x2x8xf16>) -> tensor<1x64x2x16xf16>
    %k = "onnx.Reshape"(%k4, %flat) : (tensor<1x64x2x16xf16>, tensor<3xi64>) -> tensor<1x64x32xf16>
    %v = "onnx.MatMul"(%c, %wv) : (tensor<1x64x32xf16>, tensor<32x32xf16>) -> tensor<1x64x32xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %nonpad)
        {is_causal = 1 : si64, kv_num_heads = 2 : si64, q_num_heads = 2 : si64, scale = 2.500000e-01 : f32}
        : (tensor<1x?x32xf16>, tensor<1x64x32xf16>, tensor<1x64x32xf16>, none, none, none, tensor<1xi64>)
        -> tensor<1x?x32xf16>
    return %y : tensor<1x?x32xf16>
  }
}

// -----

// ===== Negative: the Attention splits the expanded K/V into 4 heads of 8,
// not the 2 heads of 8 + 8 the expansion builds =====
module {
  func.func @main_graph(%q: tensor<1x?x32xf16>, %cache: tensor<1x1x64x40xf16>,
                        %nonpad: tensor<1xi64>)
      -> tensor<1x?x32xf16> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: hip.mla_attention
    // CHECK: hip.kv_attention
    // CHECK-NOT: hip.mla_attention
    %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %split = "onnx.Constant"() {value = dense<[32, 8]> : tensor<2xi64>} : () -> tensor<2xi64>
    %shkn = "onnx.Constant"() {value = dense<[0, 0, 2, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %shkr = "onnx.Constant"() {value = dense<[0, 0, 1, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %heads = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %flat = "onnx.Constant"() {value = dense<[0, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %wk = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<32x16xf16>} : () -> tensor<32x16xf16>
    %wv = "onnx.Constant"() {value = dense<2.000000e-02> : tensor<32x32xf16>} : () -> tensor<32x32xf16>
    %lat = "onnx.Squeeze"(%cache, %c1) : (tensor<1x1x64x40xf16>, tensor<1xi64>) -> tensor<1x64x40xf16>
    %c, %kr = "onnx.Split"(%lat, %split) {axis = -1 : si64} : (tensor<1x64x40xf16>, tensor<2xi64>) -> (tensor<1x64x32xf16>, tensor<1x64x8xf16>)
    %kn = "onnx.MatMul"(%c, %wk) : (tensor<1x64x32xf16>, tensor<32x16xf16>) -> tensor<1x64x16xf16>
    %kn4 = "onnx.Reshape"(%kn, %shkn) : (tensor<1x64x16xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %kr4 = "onnx.Reshape"(%kr, %shkr) : (tensor<1x64x8xf16>, tensor<4xi64>) -> tensor<1x64x1x8xf16>
    %krh = "onnx.Expand"(%kr4, %heads) : (tensor<1x64x1x8xf16>, tensor<4xi64>) -> tensor<1x64x2x8xf16>
    %k4 = "onnx.Concat"(%kn4, %krh) {axis = -1 : si64} : (tensor<1x64x2x8xf16>, tensor<1x64x2x8xf16>) -> tensor<1x64x2x16xf16>
    %k = "onnx.Reshape"(%k4, %flat) : (tensor<1x64x2x16xf16>, tensor<3xi64>) -> tensor<1x64x32xf16>
    %v = "onnx.MatMul"(%c, %wv) : (tensor<1x64x32xf16>, tensor<32x32xf16>) -> tensor<1x64x32xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %nonpad)
        {is_causal = 1 : si64, kv_num_heads = 4 : si64, q_num_heads = 4 : si64, scale = 2.500000e-01 : f32}
        : (tensor<1x?x32xf16>, tensor<1x64x32xf16>, tensor<1x64x32xf16>, none, none, none, tensor<1xi64>)
        -> tensor<1x?x32xf16>
    return %y : tensor<1x?x32xf16>
  }
}
