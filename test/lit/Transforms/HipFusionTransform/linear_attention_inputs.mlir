// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// A Qwen3.5 gated-delta-net layer prepares its linear-attention inputs with
// a dozen elementwise ops: q/k slices of the conv output, per-head RMS norms
// with a splat gamma, a q scale, a softplus decay and a sigmoid beta.
// hip-fusion-transform folds all of them into hip.linear_attention.
// Shapes: Hq = Nk = 2, Hkv = 4, dk = dv = 8 (packed width 2*8 + 2*8 + 4*8).

// CHECK-LABEL: func.func @la_inputs
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[QKV:[^:]*]]: tensor<1x?x64xf16>, %[[A:[^:]*]]: tensor<1x?x4xf16>, %[[B:[^:]*]]: tensor<1x?x4xf16>,
// CHECK-SAME: %[[BIAS:[^:]*]]: tensor<4xf32>, %[[SCALE:[^:]*]]: tensor<4xf32>, %[[S:[^:]*]]: tensor<1x4x8x8xf16>
// CHECK: %[[W:.*]] = hip.constant {value = dense<3.535160e-01> : tensor<8xf16>}
// CHECK-NOT: hip.rms_norm
// CHECK-NOT: hip.mul
// CHECK-NOT: hip.softplus
// CHECK-NOT: hip.sigmoid
// CHECK-NOT: hip.cast
// CHECK: %[[T:.*]] = tensor.dim %[[QKV]], %c1
// CHECK: %[[O:.*]] = tensor.empty(%[[T]]) : tensor<1x?x32xf16>
// CHECK: hip.linear_attention(%[[CTX]]) ins(%[[QKV]] : tensor<1x?x64xf16>)
// CHECK-SAME: past_state(%[[S]] : tensor<1x4x8x8xf16>) decay(%[[A]] : tensor<1x?x4xf16>) beta(%[[B]] : tensor<1x?x4xf16>)
// CHECK-SAME: decay_softplus(%[[BIAS]], %[[SCALE]] : tensor<4xf32>, tensor<4xf32>)
// CHECK-SAME: qk_norm(%[[W]] : tensor<8xf16>)
// CHECK-SAME: outs(%[[O]],
// CHECK-SAME: beta_sigmoid
// CHECK-SAME: q_scale = 2.500000e-01 : f32
// CHECK-SAME: qk_norm_epsilon = 9.99999997E-7 : f32
// CHECK-NOT: hip.
// CHECK: return
func.func @la_inputs(%ctx: !hip.context, %qkv: tensor<1x?x64xf16>,
    %a: tensor<1x?x4xf16>, %braw: tensor<1x?x4xf16>, %dt_bias: tensor<4xf32>,
    %neg_exp_a: tensor<4xf32>, %past: tensor<1x4x8x8xf16>)
    -> (tensor<1x?x32xf16>, tensor<1x4x8x8xf16>) {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %qkv, %c1 : tensor<1x?x64xf16>
  %q = tensor.extract_slice %qkv[0, 0, 0] [1, %t, 16] [1, 1, 1] : tensor<1x?x64xf16> to tensor<1x?x16xf16>
  %k = tensor.extract_slice %qkv[0, 0, 16] [1, %t, 16] [1, 1, 1] : tensor<1x?x64xf16> to tensor<1x?x16xf16>
  %v = tensor.extract_slice %qkv[0, 0, 32] [1, %t, 32] [1, 1, 1] : tensor<1x?x64xf16> to tensor<1x?x32xf16>
  %gamma = hip.constant {value = dense<3.535160e-01> : tensor<8xf16>} : tensor<8xf16>
  %qscale = hip.constant {value = dense<2.500000e-01> : tensor<1xf16>} : tensor<1xf16>

  %qe = tensor.expand_shape %q [[0], [1], [2, 3]] output_shape [1, %t, 2, 8] : tensor<1x?x16xf16> into tensor<1x?x2x8xf16>
  %eq = tensor.empty(%t) : tensor<1x?x2x8xf16>
  %qn = hip.rms_norm(%ctx) ins(%qe, %gamma : tensor<1x?x2x8xf16>, tensor<8xf16>) outs(%eq : tensor<1x?x2x8xf16>)
      {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64} : tensor<1x?x2x8xf16>
  %qc = tensor.collapse_shape %qn [[0], [1], [2, 3]] : tensor<1x?x2x8xf16> into tensor<1x?x16xf16>
  %tq = tensor.dim %qc, %c1 : tensor<1x?x16xf16>
  %eqs = tensor.empty(%tq) : tensor<1x?x16xf16>
  %qs = hip.mul(%ctx) ins(%qc, %qscale : tensor<1x?x16xf16>, tensor<1xf16>) outs(%eqs : tensor<1x?x16xf16>) -> tensor<1x?x16xf16>

  %ke = tensor.expand_shape %k [[0], [1], [2, 3]] output_shape [1, %t, 2, 8] : tensor<1x?x16xf16> into tensor<1x?x2x8xf16>
  %ek = tensor.empty(%t) : tensor<1x?x2x8xf16>
  %kn = hip.rms_norm(%ctx) ins(%ke, %gamma : tensor<1x?x2x8xf16>, tensor<8xf16>) outs(%ek : tensor<1x?x2x8xf16>)
      {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64} : tensor<1x?x2x8xf16>
  %kc = tensor.collapse_shape %kn [[0], [1], [2, 3]] : tensor<1x?x2x8xf16> into tensor<1x?x16xf16>

  %eb = tensor.empty(%t) : tensor<1x?x4xf16>
  %beta = hip.sigmoid(%ctx) ins(%braw : tensor<1x?x4xf16>) outs(%eb : tensor<1x?x4xf16>) : tensor<1x?x4xf16>

  %e32a = tensor.empty(%t) : tensor<1x?x4xf32>
  %a32 = hip.cast(%ctx) ins(%a : tensor<1x?x4xf16>) outs(%e32a : tensor<1x?x4xf32>) {to = 1 : i64} : tensor<1x?x4xf32>
  %e32b = tensor.empty(%t) : tensor<1x?x4xf32>
  %ab = hip.add(%ctx) ins(%a32, %dt_bias : tensor<1x?x4xf32>, tensor<4xf32>) outs(%e32b : tensor<1x?x4xf32>) -> tensor<1x?x4xf32>
  %e32c = tensor.empty(%t) : tensor<1x?x4xf32>
  %sp = hip.softplus(%ctx) ins(%ab : tensor<1x?x4xf32>) outs(%e32c : tensor<1x?x4xf32>) : tensor<1x?x4xf32>
  %e32d = tensor.empty(%t) : tensor<1x?x4xf32>
  %g32 = hip.mul(%ctx) ins(%neg_exp_a, %sp : tensor<4xf32>, tensor<1x?x4xf32>) outs(%e32d : tensor<1x?x4xf32>) -> tensor<1x?x4xf32>
  %eg = tensor.empty(%t) : tensor<1x?x4xf16>
  %g = hip.cast(%ctx) ins(%g32 : tensor<1x?x4xf32>) outs(%eg : tensor<1x?x4xf16>) {to = 10 : i64} : tensor<1x?x4xf16>

  %to = tensor.dim %qs, %c1 : tensor<1x?x16xf16>
  %eo = tensor.empty(%to) : tensor<1x?x32xf16>
  %es = tensor.empty() : tensor<1x4x8x8xf16>
  %o:2 = hip.linear_attention(%ctx) ins(%qs, %kc, %v : tensor<1x?x16xf16>, tensor<1x?x16xf16>, tensor<1x?x32xf16>)
      past_state(%past : tensor<1x4x8x8xf16>) decay(%g : tensor<1x?x4xf16>) beta(%beta : tensor<1x?x4xf16>)
      outs(%eo, %es : tensor<1x?x32xf16>, tensor<1x4x8x8xf16>)
      {chunk_size = 64 : i64, kv_num_heads = 4 : i64, q_num_heads = 2 : i64, scale = 1.000000e+00 : f32}
      : tensor<1x?x32xf16>, tensor<1x4x8x8xf16>
  return %o#0, %o#1 : tensor<1x?x32xf16>, tensor<1x4x8x8xf16>
}

// -----

// q and k normalized with different gammas: the norms stay, and with them
// the unpacked q/k/v; v alone is still a plain slice.

// CHECK-LABEL: func.func @la_qk_norm_mismatch
// CHECK-COUNT-2: hip.rms_norm
// CHECK: hip.linear_attention(%{{.*}}) ins(%{{[^,]*}}, %{{[^,]*}}, %{{[^:]*}} :
// CHECK-NOT: qk_norm
// CHECK: return
func.func @la_qk_norm_mismatch(%ctx: !hip.context, %qkv: tensor<1x?x64xf16>,
    %past: tensor<1x4x8x8xf16>) -> tensor<1x?x32xf16> {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %qkv, %c1 : tensor<1x?x64xf16>
  %q = tensor.extract_slice %qkv[0, 0, 0] [1, %t, 16] [1, 1, 1] : tensor<1x?x64xf16> to tensor<1x?x16xf16>
  %k = tensor.extract_slice %qkv[0, 0, 16] [1, %t, 16] [1, 1, 1] : tensor<1x?x64xf16> to tensor<1x?x16xf16>
  %v = tensor.extract_slice %qkv[0, 0, 32] [1, %t, 32] [1, 1, 1] : tensor<1x?x64xf16> to tensor<1x?x32xf16>
  %gq = hip.constant {value = dense<3.535160e-01> : tensor<8xf16>} : tensor<8xf16>
  %gk = hip.constant {value = dense<5.000000e-01> : tensor<8xf16>} : tensor<8xf16>
  %qe = tensor.expand_shape %q [[0], [1], [2, 3]] output_shape [1, %t, 2, 8] : tensor<1x?x16xf16> into tensor<1x?x2x8xf16>
  %eq = tensor.empty(%t) : tensor<1x?x2x8xf16>
  %qn = hip.rms_norm(%ctx) ins(%qe, %gq : tensor<1x?x2x8xf16>, tensor<8xf16>) outs(%eq : tensor<1x?x2x8xf16>)
      {axis = -1 : i64, epsilon = 0.000000e+00 : f32, stash_type = 1 : i64} : tensor<1x?x2x8xf16>
  %qc = tensor.collapse_shape %qn [[0], [1], [2, 3]] : tensor<1x?x2x8xf16> into tensor<1x?x16xf16>
  %ke = tensor.expand_shape %k [[0], [1], [2, 3]] output_shape [1, %t, 2, 8] : tensor<1x?x16xf16> into tensor<1x?x2x8xf16>
  %ek = tensor.empty(%t) : tensor<1x?x2x8xf16>
  %kn = hip.rms_norm(%ctx) ins(%ke, %gk : tensor<1x?x2x8xf16>, tensor<8xf16>) outs(%ek : tensor<1x?x2x8xf16>)
      {axis = -1 : i64, epsilon = 0.000000e+00 : f32, stash_type = 1 : i64} : tensor<1x?x2x8xf16>
  %kc = tensor.collapse_shape %kn [[0], [1], [2, 3]] : tensor<1x?x2x8xf16> into tensor<1x?x16xf16>
  %eo = tensor.empty(%t) : tensor<1x?x32xf16>
  %es = tensor.empty() : tensor<1x4x8x8xf16>
  %o:2 = hip.linear_attention(%ctx) ins(%qc, %kc, %v : tensor<1x?x16xf16>, tensor<1x?x16xf16>, tensor<1x?x32xf16>)
      past_state(%past : tensor<1x4x8x8xf16>)
      outs(%eo, %es : tensor<1x?x32xf16>, tensor<1x4x8x8xf16>)
      {kv_num_heads = 4 : i64, q_num_heads = 2 : i64, update_rule = "linear"}
      : tensor<1x?x32xf16>, tensor<1x4x8x8xf16>
  return %o#0 : tensor<1x?x32xf16>
}

// -----

// v does not follow k in the source: q/k/v stay separate.

// CHECK-LABEL: func.func @la_not_packed
// CHECK: %[[Q:.*]] = tensor.extract_slice
// CHECK: %[[K:.*]] = tensor.extract_slice
// CHECK: %[[V:.*]] = tensor.extract_slice
// CHECK: hip.linear_attention(%{{.*}}) ins(%[[Q]], %[[K]], %[[V]] :
func.func @la_not_packed(%ctx: !hip.context, %qkv: tensor<1x?x96xf16>,
    %past: tensor<1x4x8x8xf16>) -> tensor<1x?x32xf16> {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %qkv, %c1 : tensor<1x?x96xf16>
  %q = tensor.extract_slice %qkv[0, 0, 0] [1, %t, 16] [1, 1, 1] : tensor<1x?x96xf16> to tensor<1x?x16xf16>
  %k = tensor.extract_slice %qkv[0, 0, 16] [1, %t, 16] [1, 1, 1] : tensor<1x?x96xf16> to tensor<1x?x16xf16>
  %v = tensor.extract_slice %qkv[0, 0, 64] [1, %t, 32] [1, 1, 1] : tensor<1x?x96xf16> to tensor<1x?x32xf16>
  %eo = tensor.empty(%t) : tensor<1x?x32xf16>
  %es = tensor.empty() : tensor<1x4x8x8xf16>
  %o:2 = hip.linear_attention(%ctx) ins(%q, %k, %v : tensor<1x?x16xf16>, tensor<1x?x16xf16>, tensor<1x?x32xf16>)
      past_state(%past : tensor<1x4x8x8xf16>)
      outs(%eo, %es : tensor<1x?x32xf16>, tensor<1x4x8x8xf16>)
      {kv_num_heads = 4 : i64, q_num_heads = 2 : i64, update_rule = "linear"}
      : tensor<1x?x32xf16>, tensor<1x4x8x8xf16>
  return %o#0 : tensor<1x?x32xf16>
}

// -----

// q|k|v are contiguous, but the state says dv = 4: v (32 wide) is not
// Hkv * dv, so the packed op would split the source differently.

// CHECK-LABEL: func.func @la_packed_width_mismatch
// CHECK: %[[Q:.*]] = tensor.extract_slice
// CHECK: %[[K:.*]] = tensor.extract_slice
// CHECK: %[[V:.*]] = tensor.extract_slice
// CHECK: hip.linear_attention(%{{.*}}) ins(%[[Q]], %[[K]], %[[V]] :
func.func @la_packed_width_mismatch(%ctx: !hip.context, %qkv: tensor<1x?x64xf16>,
    %past: tensor<1x4x8x4xf16>) -> tensor<1x?x16xf16> {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %qkv, %c1 : tensor<1x?x64xf16>
  %q = tensor.extract_slice %qkv[0, 0, 0] [1, %t, 16] [1, 1, 1] : tensor<1x?x64xf16> to tensor<1x?x16xf16>
  %k = tensor.extract_slice %qkv[0, 0, 16] [1, %t, 16] [1, 1, 1] : tensor<1x?x64xf16> to tensor<1x?x16xf16>
  %v = tensor.extract_slice %qkv[0, 0, 32] [1, %t, 32] [1, 1, 1] : tensor<1x?x64xf16> to tensor<1x?x32xf16>
  %eo = tensor.empty(%t) : tensor<1x?x16xf16>
  %es = tensor.empty() : tensor<1x4x8x4xf16>
  %o:2 = hip.linear_attention(%ctx) ins(%q, %k, %v : tensor<1x?x16xf16>, tensor<1x?x16xf16>, tensor<1x?x32xf16>)
      past_state(%past : tensor<1x4x8x4xf16>)
      outs(%eo, %es : tensor<1x?x16xf16>, tensor<1x4x8x4xf16>)
      {kv_num_heads = 4 : i64, q_num_heads = 2 : i64, update_rule = "linear"}
      : tensor<1x?x16xf16>, tensor<1x4x8x4xf16>
  return %o#0 : tensor<1x?x16xf16>
}

// -----

// Odd head sizes (dk = dv = 7): the preparation kernel reads half2 pairs,
// so neither the packed source nor the raw beta is folded in.

// CHECK-LABEL: func.func @la_odd_heads
// CHECK: %[[Q:.*]] = tensor.extract_slice
// CHECK: %[[K:.*]] = tensor.extract_slice
// CHECK: %[[V:.*]] = tensor.extract_slice
// CHECK: %[[BETA:.*]] = hip.sigmoid
// CHECK: hip.linear_attention(%{{.*}}) ins(%[[Q]], %[[K]], %[[V]] :
// CHECK-SAME: beta(%[[BETA]] :
// CHECK-NOT: beta_sigmoid
// CHECK: return
func.func @la_odd_heads(%ctx: !hip.context, %qkv: tensor<1x?x56xf16>,
    %braw: tensor<1x?x4xf16>, %past: tensor<1x4x7x7xf16>) -> tensor<1x?x28xf16> {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %qkv, %c1 : tensor<1x?x56xf16>
  %q = tensor.extract_slice %qkv[0, 0, 0] [1, %t, 14] [1, 1, 1] : tensor<1x?x56xf16> to tensor<1x?x14xf16>
  %k = tensor.extract_slice %qkv[0, 0, 14] [1, %t, 14] [1, 1, 1] : tensor<1x?x56xf16> to tensor<1x?x14xf16>
  %v = tensor.extract_slice %qkv[0, 0, 28] [1, %t, 28] [1, 1, 1] : tensor<1x?x56xf16> to tensor<1x?x28xf16>
  %eb = tensor.empty(%t) : tensor<1x?x4xf16>
  %beta = hip.sigmoid(%ctx) ins(%braw : tensor<1x?x4xf16>) outs(%eb : tensor<1x?x4xf16>) : tensor<1x?x4xf16>
  %eo = tensor.empty(%t) : tensor<1x?x28xf16>
  %es = tensor.empty() : tensor<1x4x7x7xf16>
  %o:2 = hip.linear_attention(%ctx) ins(%q, %k, %v : tensor<1x?x14xf16>, tensor<1x?x14xf16>, tensor<1x?x28xf16>)
      past_state(%past : tensor<1x4x7x7xf16>) beta(%beta : tensor<1x?x4xf16>)
      outs(%eo, %es : tensor<1x?x28xf16>, tensor<1x4x7x7xf16>)
      {kv_num_heads = 4 : i64, q_num_heads = 2 : i64, update_rule = "delta"}
      : tensor<1x?x28xf16>, tensor<1x4x7x7xf16>
  return %o#0 : tensor<1x?x28xf16>
}

// -----

// The sigmoid has another user: beta stays prepared outside.

// CHECK-LABEL: func.func @la_beta_shared
// CHECK: %[[BETA:.*]] = hip.sigmoid
// CHECK: hip.linear_attention
// CHECK-SAME: beta(%[[BETA]] :
// CHECK-NOT: beta_sigmoid
// CHECK: return
func.func @la_beta_shared(%ctx: !hip.context, %q: tensor<1x?x16xf16>,
    %k: tensor<1x?x16xf16>, %v: tensor<1x?x32xf16>, %braw: tensor<1x?x4xf16>,
    %past: tensor<1x4x8x8xf16>) -> (tensor<1x?x32xf16>, tensor<1x?x4xf16>) {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %q, %c1 : tensor<1x?x16xf16>
  %eb = tensor.empty(%t) : tensor<1x?x4xf16>
  %beta = hip.sigmoid(%ctx) ins(%braw : tensor<1x?x4xf16>) outs(%eb : tensor<1x?x4xf16>) : tensor<1x?x4xf16>
  %eo = tensor.empty(%t) : tensor<1x?x32xf16>
  %es = tensor.empty() : tensor<1x4x8x8xf16>
  %o:2 = hip.linear_attention(%ctx) ins(%q, %k, %v : tensor<1x?x16xf16>, tensor<1x?x16xf16>, tensor<1x?x32xf16>)
      past_state(%past : tensor<1x4x8x8xf16>) beta(%beta : tensor<1x?x4xf16>)
      outs(%eo, %es : tensor<1x?x32xf16>, tensor<1x4x8x8xf16>)
      {kv_num_heads = 4 : i64, q_num_heads = 2 : i64, update_rule = "delta"}
      : tensor<1x?x32xf16>, tensor<1x4x8x8xf16>
  return %o#0, %beta : tensor<1x?x32xf16>, tensor<1x?x4xf16>
}
