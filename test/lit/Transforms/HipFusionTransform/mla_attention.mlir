// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// A hip.kv_attention whose keys and values are expanded from a latent cache
// [B, 1, T, L + R] becomes hip.mla_attention on the cache:
//
//   lat = collapse_shape(cache); c = lat[..., :L]; kr = lat[..., L:]
//   K   = transpose(concat(split_heads(c @ Wk), broadcast_heads(kr)))
//   V   = transpose(split_heads(c @ Wv))
//
// with constant fp16 or int4 Wk/Wv. The slices must cover every cache row,
// otherwise the fused op would read rows the expansion never produced.
// Geometry: H = 2, L = 32, R = 8, dn = 8, dv = 16, 64 cache rows.

// CHECK-LABEL: func.func @fp16
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[Q:[^:]*]]: tensor<1x?x32xf16>, %[[CACHE:[^:]*]]: tensor<1x1x64x40xf16>, %[[LEN:[^:]*]]: tensor<1xi64>)
// CHECK-DAG: %[[WK:.*]] = hip.constant {value = dense<1.000210e-02> : tensor<32x16xf16>}
// CHECK-DAG: %[[WV:.*]] = hip.constant {value = dense<2.000430e-02> : tensor<32x32xf16>}
// CHECK: %[[Y:.*]] = hip.mla_attention(%[[CTX]]) ins(%[[Q]], %[[CACHE]], %[[WK]], %[[WV]] : tensor<1x?x32xf16>, tensor<1x1x64x40xf16>, tensor<32x16xf16>, tensor<32x32xf16>)
// CHECK-SAME: kv_seqlen(%[[LEN]] : tensor<1xi64>)
// CHECK-SAME: {kv_lora_rank = 32 : i64, num_heads = 2 : i64, qk_nope_dim = 8 : i64, qk_rope_dim = 8 : i64, scale = 2.500000e-01 : f32, v_dim = 16 : i64}
// CHECK-NOT: hip.kv_attention
// CHECK-NOT: hip.matmul
// CHECK-NOT: hip.transpose
// CHECK-NOT: tensor.insert_slice
// CHECK: return %[[Y]]
func.func @fp16(%ctx: !hip.context, %q: tensor<1x?x32xf16>,
                %cache: tensor<1x1x64x40xf16>, %len: tensor<1xi64>)
    -> tensor<1x?x32xf16> {
  %c1 = arith.constant 1 : index
  %heads = hip.constant {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : tensor<4xi64>
  %wk = hip.constant {value = dense<1.000210e-02> : tensor<32x16xf16>} : tensor<32x16xf16>
  %wv = hip.constant {value = dense<2.000430e-02> : tensor<32x32xf16>} : tensor<32x32xf16>
  %lat = tensor.collapse_shape %cache [[0], [1, 2], [3]] : tensor<1x1x64x40xf16> into tensor<1x64x40xf16>
  %c = tensor.extract_slice %lat[0, 0, 0] [1, 64, 32] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x64x32xf16>
  %kr = tensor.extract_slice %lat[0, 0, 32] [1, 64, 8] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x64x8xf16>
  %e0 = tensor.empty() : tensor<1x64x16xf16>
  %kn = hip.matmul(%ctx) ins(%c, %wk : tensor<1x64x32xf16>, tensor<32x16xf16>) outs(%e0 : tensor<1x64x16xf16>) : tensor<1x64x16xf16>
  %kn4 = tensor.expand_shape %kn [[0], [1], [2, 3]] output_shape [1, 64, 2, 8] : tensor<1x64x16xf16> into tensor<1x64x2x8xf16>
  %kr4 = tensor.expand_shape %kr [[0], [1], [2, 3]] output_shape [1, 64, 1, 8] : tensor<1x64x8xf16> into tensor<1x64x1x8xf16>
  %e1 = tensor.empty() : tensor<1x64x2x8xf16>
  %krh = hip.expand(%ctx) ins(%kr4, %heads : tensor<1x64x1x8xf16>, tensor<4xi64>) outs(%e1 : tensor<1x64x2x8xf16>) : tensor<1x64x2x8xf16>
  %e2 = tensor.empty() : tensor<1x64x2x16xf16>
  %k0 = tensor.insert_slice %kn4 into %e2[0, 0, 0, 0] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %k4 = tensor.insert_slice %krh into %k0[0, 0, 0, 8] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %e3 = tensor.empty() : tensor<1x64x32xf16>
  %v = hip.matmul(%ctx) ins(%c, %wv : tensor<1x64x32xf16>, tensor<32x32xf16>) outs(%e3 : tensor<1x64x32xf16>) : tensor<1x64x32xf16>
  %v4 = tensor.expand_shape %v [[0], [1], [2, 3]] output_shape [1, 64, 2, 16] : tensor<1x64x32xf16> into tensor<1x64x2x16xf16>
  %e4 = tensor.empty() : tensor<1x2x64x16xf16>
  %kt = hip.transpose(%ctx) ins(%k4 : tensor<1x64x2x16xf16>) outs(%e4 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %e5 = tensor.empty() : tensor<1x2x64x16xf16>
  %vt = hip.transpose(%ctx) ins(%v4 : tensor<1x64x2x16xf16>) outs(%e5 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %s = tensor.dim %q, %c1 : tensor<1x?x32xf16>
  %e6 = tensor.empty(%s) : tensor<1x?x32xf16>
  %y = hip.kv_attention(%ctx) ins(%q, %kt, %vt : tensor<1x?x32xf16>, tensor<1x2x64x16xf16>, tensor<1x2x64x16xf16>) kv_seqlen(%len : tensor<1xi64>) outs(%e6 : tensor<1x?x32xf16>) {kv_num_heads = 2 : i64, num_heads = 2 : i64, scale = 2.500000e-01 : f32} : tensor<1x?x32xf16>
  return %y : tensor<1x?x32xf16>
}

// -----

// int4 up-projections (block 16), rope part inserted first, and no
// kv_seqlen: every cache row is a valid key.
// CHECK-LABEL: func.func @int4_rope_first
// CHECK: hip.mla_attention(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x?x32xf16>, tensor<1x1x64x40xf16>, tensor<16x2x8xui8>, tensor<32x2x8xui8>)
// CHECK-SAME: k_b_scales(%{{.*}} : tensor<16x2xf16>) v_b_scales(%{{.*}} : tensor<32x2xf16>) outs(
// CHECK-SAME: block_size = 16 : i64
// CHECK-NOT: hip.kv_attention
// CHECK-NOT: hip.matmul_nbits
func.func @int4_rope_first(%ctx: !hip.context, %q: tensor<1x?x32xf16>,
                           %cache: tensor<1x1x64x40xf16>)
    -> tensor<1x?x32xf16> {
  %c1 = arith.constant 1 : index
  %heads = hip.constant {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : tensor<4xi64>
  %wk = hip.constant {value = dense<136> : tensor<16x2x8xui8>} : tensor<16x2x8xui8>
  %sk = hip.constant {value = dense<1.000210e-02> : tensor<16x2xf16>} : tensor<16x2xf16>
  %wv = hip.constant {value = dense<137> : tensor<32x2x8xui8>} : tensor<32x2x8xui8>
  %sv = hip.constant {value = dense<2.000430e-02> : tensor<32x2xf16>} : tensor<32x2xf16>
  %lat = tensor.collapse_shape %cache [[0], [1, 2], [3]] : tensor<1x1x64x40xf16> into tensor<1x64x40xf16>
  %c = tensor.extract_slice %lat[0, 0, 0] [1, 64, 32] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x64x32xf16>
  %kr = tensor.extract_slice %lat[0, 0, 32] [1, 64, 8] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x64x8xf16>
  %e0 = tensor.empty() : tensor<1x64x16xf16>
  %kn = hip.matmul_nbits(%ctx) ins(%c, %wk, %sk : tensor<1x64x32xf16>, tensor<16x2x8xui8>, tensor<16x2xf16>) outs(%e0 : tensor<1x64x16xf16>) {K = 32 : i64, N = 16 : i64, accuracy_level = 0 : i64, bits = 4 : i64, block_size = 16 : i64, zp_elem_size = 2 : i64} : tensor<1x64x16xf16>
  %kn4 = tensor.expand_shape %kn [[0], [1], [2, 3]] output_shape [1, 64, 2, 8] : tensor<1x64x16xf16> into tensor<1x64x2x8xf16>
  %kr4 = tensor.expand_shape %kr [[0], [1], [2, 3]] output_shape [1, 64, 1, 8] : tensor<1x64x8xf16> into tensor<1x64x1x8xf16>
  %e1 = tensor.empty() : tensor<1x64x2x8xf16>
  %krh = hip.expand(%ctx) ins(%kr4, %heads : tensor<1x64x1x8xf16>, tensor<4xi64>) outs(%e1 : tensor<1x64x2x8xf16>) : tensor<1x64x2x8xf16>
  %e2 = tensor.empty() : tensor<1x64x2x16xf16>
  %k0 = tensor.insert_slice %krh into %e2[0, 0, 0, 8] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %k4 = tensor.insert_slice %kn4 into %k0[0, 0, 0, 0] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %e3 = tensor.empty() : tensor<1x64x32xf16>
  %v = hip.matmul_nbits(%ctx) ins(%c, %wv, %sv : tensor<1x64x32xf16>, tensor<32x2x8xui8>, tensor<32x2xf16>) outs(%e3 : tensor<1x64x32xf16>) {K = 32 : i64, N = 32 : i64, accuracy_level = 0 : i64, bits = 4 : i64, block_size = 16 : i64, zp_elem_size = 2 : i64} : tensor<1x64x32xf16>
  %v4 = tensor.expand_shape %v [[0], [1], [2, 3]] output_shape [1, 64, 2, 16] : tensor<1x64x32xf16> into tensor<1x64x2x16xf16>
  %e4 = tensor.empty() : tensor<1x2x64x16xf16>
  %kt = hip.transpose(%ctx) ins(%k4 : tensor<1x64x2x16xf16>) outs(%e4 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %e5 = tensor.empty() : tensor<1x2x64x16xf16>
  %vt = hip.transpose(%ctx) ins(%v4 : tensor<1x64x2x16xf16>) outs(%e5 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %s = tensor.dim %q, %c1 : tensor<1x?x32xf16>
  %e6 = tensor.empty(%s) : tensor<1x?x32xf16>
  %y = hip.kv_attention(%ctx) ins(%q, %kt, %vt : tensor<1x?x32xf16>, tensor<1x2x64x16xf16>, tensor<1x2x64x16xf16>) outs(%e6 : tensor<1x?x32xf16>) {kv_num_heads = 2 : i64, num_heads = 2 : i64, scale = 2.500000e-01 : f32} : tensor<1x?x32xf16>
  return %y : tensor<1x?x32xf16>
}

// -----

// Dynamic cache length: every slice size is a tensor.dim that traces back to
// the cache row count, and the keys pass through a collapse_shape /
// expand_shape round trip before the transpose.
// CHECK-LABEL: func.func @dynamic_rows
// CHECK-SAME: %[[CACHE:[^:]*]]: tensor<1x1x?x40xf16>
// CHECK: hip.mla_attention(%{{.*}}) ins(%{{.*}}, %[[CACHE]], %{{.*}}, %{{.*}} : tensor<1x?x32xf16>, tensor<1x1x?x40xf16>, tensor<32x16xf16>, tensor<32x32xf16>)
// CHECK-NOT: hip.kv_attention
func.func @dynamic_rows(%ctx: !hip.context, %q: tensor<1x?x32xf16>,
                        %cache: tensor<1x1x?x40xf16>, %len: tensor<1xi64>)
    -> tensor<1x?x32xf16> {
  %c1 = arith.constant 1 : index
  %heads = hip.constant {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : tensor<4xi64>
  %wk = hip.constant {value = dense<1.000210e-02> : tensor<32x16xf16>} : tensor<32x16xf16>
  %wv = hip.constant {value = dense<2.000430e-02> : tensor<32x32xf16>} : tensor<32x32xf16>
  %lat = tensor.collapse_shape %cache [[0], [1, 2], [3]] : tensor<1x1x?x40xf16> into tensor<1x?x40xf16>
  %t0 = tensor.dim %lat, %c1 : tensor<1x?x40xf16>
  %c = tensor.extract_slice %lat[0, 0, 0] [1, %t0, 32] [1, 1, 1] : tensor<1x?x40xf16> to tensor<1x?x32xf16>
  %t1 = tensor.dim %lat, %c1 : tensor<1x?x40xf16>
  %kr = tensor.extract_slice %lat[0, 0, 32] [1, %t1, 8] [1, 1, 1] : tensor<1x?x40xf16> to tensor<1x?x8xf16>
  %t2 = tensor.dim %c, %c1 : tensor<1x?x32xf16>
  %e0 = tensor.empty(%t2) : tensor<1x?x16xf16>
  %kn = hip.matmul(%ctx) ins(%c, %wk : tensor<1x?x32xf16>, tensor<32x16xf16>) outs(%e0 : tensor<1x?x16xf16>) : tensor<1x?x16xf16>
  %t3 = tensor.dim %kn, %c1 : tensor<1x?x16xf16>
  %kn4 = tensor.expand_shape %kn [[0], [1], [2, 3]] output_shape [1, %t3, 2, 8] : tensor<1x?x16xf16> into tensor<1x?x2x8xf16>
  %t4 = tensor.dim %kr, %c1 : tensor<1x?x8xf16>
  %kr4 = tensor.expand_shape %kr [[0], [1], [2, 3]] output_shape [1, %t4, 1, 8] : tensor<1x?x8xf16> into tensor<1x?x1x8xf16>
  %t5 = tensor.dim %kr4, %c1 : tensor<1x?x1x8xf16>
  %e1 = tensor.empty(%t5) : tensor<1x?x2x8xf16>
  %krh = hip.expand(%ctx) ins(%kr4, %heads : tensor<1x?x1x8xf16>, tensor<4xi64>) outs(%e1 : tensor<1x?x2x8xf16>) : tensor<1x?x2x8xf16>
  %t6 = tensor.dim %kn4, %c1 : tensor<1x?x2x8xf16>
  %e2 = tensor.empty(%t6) : tensor<1x?x2x16xf16>
  %t7 = tensor.dim %kn4, %c1 : tensor<1x?x2x8xf16>
  %k0 = tensor.insert_slice %kn4 into %e2[0, 0, 0, 0] [1, %t7, 2, 8] [1, 1, 1, 1] : tensor<1x?x2x8xf16> into tensor<1x?x2x16xf16>
  %t8 = tensor.dim %krh, %c1 : tensor<1x?x2x8xf16>
  %k4 = tensor.insert_slice %krh into %k0[0, 0, 0, 8] [1, %t8, 2, 8] [1, 1, 1, 1] : tensor<1x?x2x8xf16> into tensor<1x?x2x16xf16>
  %k3 = tensor.collapse_shape %k4 [[0], [1], [2, 3]] : tensor<1x?x2x16xf16> into tensor<1x?x32xf16>
  %t9 = tensor.dim %k3, %c1 : tensor<1x?x32xf16>
  %k4b = tensor.expand_shape %k3 [[0], [1], [2, 3]] output_shape [1, %t9, 2, 16] : tensor<1x?x32xf16> into tensor<1x?x2x16xf16>
  %t10 = tensor.dim %c, %c1 : tensor<1x?x32xf16>
  %e3 = tensor.empty(%t10) : tensor<1x?x32xf16>
  %v = hip.matmul(%ctx) ins(%c, %wv : tensor<1x?x32xf16>, tensor<32x32xf16>) outs(%e3 : tensor<1x?x32xf16>) : tensor<1x?x32xf16>
  %t11 = tensor.dim %v, %c1 : tensor<1x?x32xf16>
  %v4 = tensor.expand_shape %v [[0], [1], [2, 3]] output_shape [1, %t11, 2, 16] : tensor<1x?x32xf16> into tensor<1x?x2x16xf16>
  %t12 = tensor.dim %k4b, %c1 : tensor<1x?x2x16xf16>
  %e4 = tensor.empty(%t12) : tensor<1x2x?x16xf16>
  %kt = hip.transpose(%ctx) ins(%k4b : tensor<1x?x2x16xf16>) outs(%e4 : tensor<1x2x?x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x?x16xf16>
  %t13 = tensor.dim %v4, %c1 : tensor<1x?x2x16xf16>
  %e5 = tensor.empty(%t13) : tensor<1x2x?x16xf16>
  %vt = hip.transpose(%ctx) ins(%v4 : tensor<1x?x2x16xf16>) outs(%e5 : tensor<1x2x?x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x?x16xf16>
  %s = tensor.dim %q, %c1 : tensor<1x?x32xf16>
  %e6 = tensor.empty(%s) : tensor<1x?x32xf16>
  %y = hip.kv_attention(%ctx) ins(%q, %kt, %vt : tensor<1x?x32xf16>, tensor<1x2x?x16xf16>, tensor<1x2x?x16xf16>) kv_seqlen(%len : tensor<1xi64>) outs(%e6 : tensor<1x?x32xf16>) {kv_num_heads = 2 : i64, num_heads = 2 : i64, scale = 2.500000e-01 : f32} : tensor<1x?x32xf16>
  return %y : tensor<1x?x32xf16>
}

// -----

// Negative: the latent slices cover cache rows 0..32 of 64, so the expanded
// keys are not the cache.
// CHECK-LABEL: func.func @partial_rows
// CHECK-NOT: hip.mla_attention
// CHECK: hip.kv_attention
func.func @partial_rows(%ctx: !hip.context, %q: tensor<1x?x32xf16>,
                        %cache: tensor<1x1x64x40xf16>, %len: tensor<1xi64>)
    -> tensor<1x?x32xf16> {
  %c1 = arith.constant 1 : index
  %heads = hip.constant {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : tensor<4xi64>
  %wk = hip.constant {value = dense<1.000210e-02> : tensor<32x16xf16>} : tensor<32x16xf16>
  %wv = hip.constant {value = dense<2.000430e-02> : tensor<32x32xf16>} : tensor<32x32xf16>
  %lat = tensor.collapse_shape %cache [[0], [1, 2], [3]] : tensor<1x1x64x40xf16> into tensor<1x64x40xf16>
  %c = tensor.extract_slice %lat[0, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x32x32xf16>
  %kr = tensor.extract_slice %lat[0, 0, 32] [1, 32, 8] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x32x8xf16>
  %e0 = tensor.empty() : tensor<1x32x16xf16>
  %kn = hip.matmul(%ctx) ins(%c, %wk : tensor<1x32x32xf16>, tensor<32x16xf16>) outs(%e0 : tensor<1x32x16xf16>) : tensor<1x32x16xf16>
  %kn4 = tensor.expand_shape %kn [[0], [1], [2, 3]] output_shape [1, 32, 2, 8] : tensor<1x32x16xf16> into tensor<1x32x2x8xf16>
  %kr4 = tensor.expand_shape %kr [[0], [1], [2, 3]] output_shape [1, 32, 1, 8] : tensor<1x32x8xf16> into tensor<1x32x1x8xf16>
  %e1 = tensor.empty() : tensor<1x32x2x8xf16>
  %krh = hip.expand(%ctx) ins(%kr4, %heads : tensor<1x32x1x8xf16>, tensor<4xi64>) outs(%e1 : tensor<1x32x2x8xf16>) : tensor<1x32x2x8xf16>
  %e2 = tensor.empty() : tensor<1x32x2x16xf16>
  %k0 = tensor.insert_slice %kn4 into %e2[0, 0, 0, 0] [1, 32, 2, 8] [1, 1, 1, 1] : tensor<1x32x2x8xf16> into tensor<1x32x2x16xf16>
  %k4 = tensor.insert_slice %krh into %k0[0, 0, 0, 8] [1, 32, 2, 8] [1, 1, 1, 1] : tensor<1x32x2x8xf16> into tensor<1x32x2x16xf16>
  %e3 = tensor.empty() : tensor<1x32x32xf16>
  %v = hip.matmul(%ctx) ins(%c, %wv : tensor<1x32x32xf16>, tensor<32x32xf16>) outs(%e3 : tensor<1x32x32xf16>) : tensor<1x32x32xf16>
  %v4 = tensor.expand_shape %v [[0], [1], [2, 3]] output_shape [1, 32, 2, 16] : tensor<1x32x32xf16> into tensor<1x32x2x16xf16>
  %e4 = tensor.empty() : tensor<1x2x32x16xf16>
  %kt = hip.transpose(%ctx) ins(%k4 : tensor<1x32x2x16xf16>) outs(%e4 : tensor<1x2x32x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x32x16xf16>
  %e5 = tensor.empty() : tensor<1x2x32x16xf16>
  %vt = hip.transpose(%ctx) ins(%v4 : tensor<1x32x2x16xf16>) outs(%e5 : tensor<1x2x32x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x32x16xf16>
  %s = tensor.dim %q, %c1 : tensor<1x?x32xf16>
  %e6 = tensor.empty(%s) : tensor<1x?x32xf16>
  %y = hip.kv_attention(%ctx) ins(%q, %kt, %vt : tensor<1x?x32xf16>, tensor<1x2x32x16xf16>, tensor<1x2x32x16xf16>) kv_seqlen(%len : tensor<1xi64>) outs(%e6 : tensor<1x?x32xf16>) {kv_num_heads = 2 : i64, num_heads = 2 : i64, scale = 2.500000e-01 : f32} : tensor<1x?x32xf16>
  return %y : tensor<1x?x32xf16>
}

// -----

// Negative: the projected latent starts at column 8, so it overlaps the rope
// columns instead of being the cache's first L columns.
// CHECK-LABEL: func.func @latent_offset
// CHECK-NOT: hip.mla_attention
// CHECK: hip.kv_attention
func.func @latent_offset(%ctx: !hip.context, %q: tensor<1x?x32xf16>,
                         %cache: tensor<1x1x64x40xf16>, %len: tensor<1xi64>)
    -> tensor<1x?x32xf16> {
  %c1 = arith.constant 1 : index
  %heads = hip.constant {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : tensor<4xi64>
  %wk = hip.constant {value = dense<1.000210e-02> : tensor<32x16xf16>} : tensor<32x16xf16>
  %wv = hip.constant {value = dense<2.000430e-02> : tensor<32x32xf16>} : tensor<32x32xf16>
  %lat = tensor.collapse_shape %cache [[0], [1, 2], [3]] : tensor<1x1x64x40xf16> into tensor<1x64x40xf16>
  %c = tensor.extract_slice %lat[0, 0, 8] [1, 64, 32] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x64x32xf16>
  %kr = tensor.extract_slice %lat[0, 0, 32] [1, 64, 8] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x64x8xf16>
  %e0 = tensor.empty() : tensor<1x64x16xf16>
  %kn = hip.matmul(%ctx) ins(%c, %wk : tensor<1x64x32xf16>, tensor<32x16xf16>) outs(%e0 : tensor<1x64x16xf16>) : tensor<1x64x16xf16>
  %kn4 = tensor.expand_shape %kn [[0], [1], [2, 3]] output_shape [1, 64, 2, 8] : tensor<1x64x16xf16> into tensor<1x64x2x8xf16>
  %kr4 = tensor.expand_shape %kr [[0], [1], [2, 3]] output_shape [1, 64, 1, 8] : tensor<1x64x8xf16> into tensor<1x64x1x8xf16>
  %e1 = tensor.empty() : tensor<1x64x2x8xf16>
  %krh = hip.expand(%ctx) ins(%kr4, %heads : tensor<1x64x1x8xf16>, tensor<4xi64>) outs(%e1 : tensor<1x64x2x8xf16>) : tensor<1x64x2x8xf16>
  %e2 = tensor.empty() : tensor<1x64x2x16xf16>
  %k0 = tensor.insert_slice %kn4 into %e2[0, 0, 0, 0] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %k4 = tensor.insert_slice %krh into %k0[0, 0, 0, 8] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %e3 = tensor.empty() : tensor<1x64x32xf16>
  %v = hip.matmul(%ctx) ins(%c, %wv : tensor<1x64x32xf16>, tensor<32x32xf16>) outs(%e3 : tensor<1x64x32xf16>) : tensor<1x64x32xf16>
  %v4 = tensor.expand_shape %v [[0], [1], [2, 3]] output_shape [1, 64, 2, 16] : tensor<1x64x32xf16> into tensor<1x64x2x16xf16>
  %e4 = tensor.empty() : tensor<1x2x64x16xf16>
  %kt = hip.transpose(%ctx) ins(%k4 : tensor<1x64x2x16xf16>) outs(%e4 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %e5 = tensor.empty() : tensor<1x2x64x16xf16>
  %vt = hip.transpose(%ctx) ins(%v4 : tensor<1x64x2x16xf16>) outs(%e5 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %s = tensor.dim %q, %c1 : tensor<1x?x32xf16>
  %e6 = tensor.empty(%s) : tensor<1x?x32xf16>
  %y = hip.kv_attention(%ctx) ins(%q, %kt, %vt : tensor<1x?x32xf16>, tensor<1x2x64x16xf16>, tensor<1x2x64x16xf16>) kv_seqlen(%len : tensor<1xi64>) outs(%e6 : tensor<1x?x32xf16>) {kv_num_heads = 2 : i64, num_heads = 2 : i64, scale = 2.500000e-01 : f32} : tensor<1x?x32xf16>
  return %y : tensor<1x?x32xf16>
}

// -----

// Negative: the rope part is already per head rather than one rope key
// broadcast over the heads.
// CHECK-LABEL: func.func @rope_not_broadcast
// CHECK-NOT: hip.mla_attention
// CHECK: hip.kv_attention
func.func @rope_not_broadcast(%ctx: !hip.context, %q: tensor<1x?x32xf16>,
                              %cache: tensor<1x1x64x40xf16>, %len: tensor<1xi64>,
                              %krh: tensor<1x64x2x8xf16>)
    -> tensor<1x?x32xf16> {
  %c1 = arith.constant 1 : index
  %wk = hip.constant {value = dense<1.000210e-02> : tensor<32x16xf16>} : tensor<32x16xf16>
  %wv = hip.constant {value = dense<2.000430e-02> : tensor<32x32xf16>} : tensor<32x32xf16>
  %lat = tensor.collapse_shape %cache [[0], [1, 2], [3]] : tensor<1x1x64x40xf16> into tensor<1x64x40xf16>
  %c = tensor.extract_slice %lat[0, 0, 0] [1, 64, 32] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x64x32xf16>
  %e0 = tensor.empty() : tensor<1x64x16xf16>
  %kn = hip.matmul(%ctx) ins(%c, %wk : tensor<1x64x32xf16>, tensor<32x16xf16>) outs(%e0 : tensor<1x64x16xf16>) : tensor<1x64x16xf16>
  %kn4 = tensor.expand_shape %kn [[0], [1], [2, 3]] output_shape [1, 64, 2, 8] : tensor<1x64x16xf16> into tensor<1x64x2x8xf16>
  %e2 = tensor.empty() : tensor<1x64x2x16xf16>
  %k0 = tensor.insert_slice %kn4 into %e2[0, 0, 0, 0] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %k4 = tensor.insert_slice %krh into %k0[0, 0, 0, 8] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %e3 = tensor.empty() : tensor<1x64x32xf16>
  %v = hip.matmul(%ctx) ins(%c, %wv : tensor<1x64x32xf16>, tensor<32x32xf16>) outs(%e3 : tensor<1x64x32xf16>) : tensor<1x64x32xf16>
  %v4 = tensor.expand_shape %v [[0], [1], [2, 3]] output_shape [1, 64, 2, 16] : tensor<1x64x32xf16> into tensor<1x64x2x16xf16>
  %e4 = tensor.empty() : tensor<1x2x64x16xf16>
  %kt = hip.transpose(%ctx) ins(%k4 : tensor<1x64x2x16xf16>) outs(%e4 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %e5 = tensor.empty() : tensor<1x2x64x16xf16>
  %vt = hip.transpose(%ctx) ins(%v4 : tensor<1x64x2x16xf16>) outs(%e5 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %s = tensor.dim %q, %c1 : tensor<1x?x32xf16>
  %e6 = tensor.empty(%s) : tensor<1x?x32xf16>
  %y = hip.kv_attention(%ctx) ins(%q, %kt, %vt : tensor<1x?x32xf16>, tensor<1x2x64x16xf16>, tensor<1x2x64x16xf16>) kv_seqlen(%len : tensor<1xi64>) outs(%e6 : tensor<1x?x32xf16>) {kv_num_heads = 2 : i64, num_heads = 2 : i64, scale = 2.500000e-01 : f32} : tensor<1x?x32xf16>
  return %y : tensor<1x?x32xf16>
}

// -----

// Negative: the key up-projection weight is a function argument, so the
// fused op has no constant to read it from at compile time.
// CHECK-LABEL: func.func @weight_not_constant
// CHECK-NOT: hip.mla_attention
// CHECK: hip.kv_attention
func.func @weight_not_constant(%ctx: !hip.context, %q: tensor<1x?x32xf16>,
                               %cache: tensor<1x1x64x40xf16>, %len: tensor<1xi64>,
                               %wk: tensor<32x16xf16>)
    -> tensor<1x?x32xf16> {
  %c1 = arith.constant 1 : index
  %heads = hip.constant {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : tensor<4xi64>
  %wv = hip.constant {value = dense<2.000430e-02> : tensor<32x32xf16>} : tensor<32x32xf16>
  %lat = tensor.collapse_shape %cache [[0], [1, 2], [3]] : tensor<1x1x64x40xf16> into tensor<1x64x40xf16>
  %c = tensor.extract_slice %lat[0, 0, 0] [1, 64, 32] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x64x32xf16>
  %kr = tensor.extract_slice %lat[0, 0, 32] [1, 64, 8] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x64x8xf16>
  %e0 = tensor.empty() : tensor<1x64x16xf16>
  %kn = hip.matmul(%ctx) ins(%c, %wk : tensor<1x64x32xf16>, tensor<32x16xf16>) outs(%e0 : tensor<1x64x16xf16>) : tensor<1x64x16xf16>
  %kn4 = tensor.expand_shape %kn [[0], [1], [2, 3]] output_shape [1, 64, 2, 8] : tensor<1x64x16xf16> into tensor<1x64x2x8xf16>
  %kr4 = tensor.expand_shape %kr [[0], [1], [2, 3]] output_shape [1, 64, 1, 8] : tensor<1x64x8xf16> into tensor<1x64x1x8xf16>
  %e1 = tensor.empty() : tensor<1x64x2x8xf16>
  %krh = hip.expand(%ctx) ins(%kr4, %heads : tensor<1x64x1x8xf16>, tensor<4xi64>) outs(%e1 : tensor<1x64x2x8xf16>) : tensor<1x64x2x8xf16>
  %e2 = tensor.empty() : tensor<1x64x2x16xf16>
  %k0 = tensor.insert_slice %kn4 into %e2[0, 0, 0, 0] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %k4 = tensor.insert_slice %krh into %k0[0, 0, 0, 8] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %e3 = tensor.empty() : tensor<1x64x32xf16>
  %v = hip.matmul(%ctx) ins(%c, %wv : tensor<1x64x32xf16>, tensor<32x32xf16>) outs(%e3 : tensor<1x64x32xf16>) : tensor<1x64x32xf16>
  %v4 = tensor.expand_shape %v [[0], [1], [2, 3]] output_shape [1, 64, 2, 16] : tensor<1x64x32xf16> into tensor<1x64x2x16xf16>
  %e4 = tensor.empty() : tensor<1x2x64x16xf16>
  %kt = hip.transpose(%ctx) ins(%k4 : tensor<1x64x2x16xf16>) outs(%e4 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %e5 = tensor.empty() : tensor<1x2x64x16xf16>
  %vt = hip.transpose(%ctx) ins(%v4 : tensor<1x64x2x16xf16>) outs(%e5 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %s = tensor.dim %q, %c1 : tensor<1x?x32xf16>
  %e6 = tensor.empty(%s) : tensor<1x?x32xf16>
  %y = hip.kv_attention(%ctx) ins(%q, %kt, %vt : tensor<1x?x32xf16>, tensor<1x2x64x16xf16>, tensor<1x2x64x16xf16>) kv_seqlen(%len : tensor<1xi64>) outs(%e6 : tensor<1x?x32xf16>) {kv_num_heads = 2 : i64, num_heads = 2 : i64, scale = 2.500000e-01 : f32} : tensor<1x?x32xf16>
  return %y : tensor<1x?x32xf16>
}

// -----

// Negative: an int4 up-projection has zero points, which the fused kernel
// does not read.
// CHECK-LABEL: func.func @zero_points
// CHECK-NOT: hip.mla_attention
// CHECK: hip.kv_attention
func.func @zero_points(%ctx: !hip.context, %q: tensor<1x?x32xf16>,
                       %cache: tensor<1x1x64x40xf16>, %len: tensor<1xi64>)
    -> tensor<1x?x32xf16> {
  %c1 = arith.constant 1 : index
  %heads = hip.constant {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : tensor<4xi64>
  %wk = hip.constant {value = dense<136> : tensor<16x2x8xui8>} : tensor<16x2x8xui8>
  %sk = hip.constant {value = dense<1.000210e-02> : tensor<16x2xf16>} : tensor<16x2xf16>
  %zk = hip.constant {value = dense<136> : tensor<16x1xui8>} : tensor<16x1xui8>
  %wv = hip.constant {value = dense<137> : tensor<32x2x8xui8>} : tensor<32x2x8xui8>
  %sv = hip.constant {value = dense<2.000430e-02> : tensor<32x2xf16>} : tensor<32x2xf16>
  %lat = tensor.collapse_shape %cache [[0], [1, 2], [3]] : tensor<1x1x64x40xf16> into tensor<1x64x40xf16>
  %c = tensor.extract_slice %lat[0, 0, 0] [1, 64, 32] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x64x32xf16>
  %kr = tensor.extract_slice %lat[0, 0, 32] [1, 64, 8] [1, 1, 1] : tensor<1x64x40xf16> to tensor<1x64x8xf16>
  %e0 = tensor.empty() : tensor<1x64x16xf16>
  %kn = hip.matmul_nbits(%ctx) ins(%c, %wk, %sk : tensor<1x64x32xf16>, tensor<16x2x8xui8>, tensor<16x2xf16>) zero_points(%zk : tensor<16x1xui8>) outs(%e0 : tensor<1x64x16xf16>) {K = 32 : i64, N = 16 : i64, accuracy_level = 0 : i64, bits = 4 : i64, block_size = 16 : i64, zp_elem_size = 1 : i64} : tensor<1x64x16xf16>
  %kn4 = tensor.expand_shape %kn [[0], [1], [2, 3]] output_shape [1, 64, 2, 8] : tensor<1x64x16xf16> into tensor<1x64x2x8xf16>
  %kr4 = tensor.expand_shape %kr [[0], [1], [2, 3]] output_shape [1, 64, 1, 8] : tensor<1x64x8xf16> into tensor<1x64x1x8xf16>
  %e1 = tensor.empty() : tensor<1x64x2x8xf16>
  %krh = hip.expand(%ctx) ins(%kr4, %heads : tensor<1x64x1x8xf16>, tensor<4xi64>) outs(%e1 : tensor<1x64x2x8xf16>) : tensor<1x64x2x8xf16>
  %e2 = tensor.empty() : tensor<1x64x2x16xf16>
  %k0 = tensor.insert_slice %kn4 into %e2[0, 0, 0, 0] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %k4 = tensor.insert_slice %krh into %k0[0, 0, 0, 8] [1, 64, 2, 8] [1, 1, 1, 1] : tensor<1x64x2x8xf16> into tensor<1x64x2x16xf16>
  %e3 = tensor.empty() : tensor<1x64x32xf16>
  %v = hip.matmul_nbits(%ctx) ins(%c, %wv, %sv : tensor<1x64x32xf16>, tensor<32x2x8xui8>, tensor<32x2xf16>) outs(%e3 : tensor<1x64x32xf16>) {K = 32 : i64, N = 32 : i64, accuracy_level = 0 : i64, bits = 4 : i64, block_size = 16 : i64, zp_elem_size = 2 : i64} : tensor<1x64x32xf16>
  %v4 = tensor.expand_shape %v [[0], [1], [2, 3]] output_shape [1, 64, 2, 16] : tensor<1x64x32xf16> into tensor<1x64x2x16xf16>
  %e4 = tensor.empty() : tensor<1x2x64x16xf16>
  %kt = hip.transpose(%ctx) ins(%k4 : tensor<1x64x2x16xf16>) outs(%e4 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %e5 = tensor.empty() : tensor<1x2x64x16xf16>
  %vt = hip.transpose(%ctx) ins(%v4 : tensor<1x64x2x16xf16>) outs(%e5 : tensor<1x2x64x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x64x16xf16>
  %s = tensor.dim %q, %c1 : tensor<1x?x32xf16>
  %e6 = tensor.empty(%s) : tensor<1x?x32xf16>
  %y = hip.kv_attention(%ctx) ins(%q, %kt, %vt : tensor<1x?x32xf16>, tensor<1x2x64x16xf16>, tensor<1x2x64x16xf16>) kv_seqlen(%len : tensor<1xi64>) outs(%e6 : tensor<1x?x32xf16>) {kv_num_heads = 2 : i64, num_heads = 2 : i64, scale = 2.500000e-01 : f32} : tensor<1x?x32xf16>
  return %y : tensor<1x?x32xf16>
}
