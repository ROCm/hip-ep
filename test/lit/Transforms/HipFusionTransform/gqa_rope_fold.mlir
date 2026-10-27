// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// A GroupQueryAttention export with do_rotary = 0 rotates Q and K through two
// RotaryEmbedding nodes. hip-fusion-transform hands that rotation to hip.gqa
// (do_rotary = 1, same position_ids / cos / sin) when it is exactly the one
// hip.gqa implements, and leaves the IR untouched otherwise.

// CHECK-LABEL: func.func @fold_decode
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[Q:[^:]*]]: tensor<1x1x64xf16>, %[[K:[^:]*]]: tensor<1x1x32xf16>, %[[V:[^:]*]]: tensor<1x1x32xf16>, %[[PK:[^:]*]]: tensor<1x2x8x16xf16>, %[[PV:[^:]*]]: tensor<1x2x8x16xf16>, %[[SL:[^:]*]]: tensor<1xi32>, %[[TOT:[^:]*]]: tensor<i32>, %[[POS:[^:]*]]: tensor<1x1xi64>, %[[COS:[^:]*]]: tensor<32x8xf16>, %[[SIN:[^:]*]]: tensor<32x8xf16>)
// CHECK-NOT: hip.rope
// CHECK: hip.gqa(%[[CTX]]) ins(%[[Q]], %[[K]], %[[V]], %[[PK]], %[[PV]], %[[SL]], %[[TOT]], %[[COS]], %[[SIN]], %[[POS]] :
// CHECK-SAME: do_rotary = 1
// CHECK-NOT: rotary_interleaved
// CHECK-NOT: hip.rope
func.func @fold_decode(%ctx: !hip.context, %q: tensor<1x1x64xf16>,
                       %k: tensor<1x1x32xf16>, %v: tensor<1x1x32xf16>,
                       %pk: tensor<1x2x8x16xf16>, %pv: tensor<1x2x8x16xf16>,
                       %sl: tensor<1xi32>, %tot: tensor<i32>,
                       %pos: tensor<1x1xi64>, %cos: tensor<32x8xf16>,
                       %sin: tensor<32x8xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>) {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %qr = hip.rope(%ctx) ins(%q, %pos, %cos, %sin : tensor<1x1x64xf16>, tensor<1x1xi64>, tensor<32x8xf16>, tensor<32x8xf16>)
        outs(%e0 : tensor<1x1x64xf16>)
        {interleaved = 0 : i64, num_heads = 4 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %kr = hip.rope(%ctx) ins(%k, %pos, %cos, %sin : tensor<1x1x32xf16>, tensor<1x1xi64>, tensor<32x8xf16>, tensor<32x8xf16>)
        outs(%e1 : tensor<1x1x32xf16>)
        {interleaved = 0 : i64, num_heads = 2 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x1x32xf16>
  %o = tensor.empty() : tensor<1x1x64xf16>
  %r:3 = hip.gqa(%ctx) ins(%qr, %kr, %v, %pk, %pv, %sl, %tot : tensor<1x1x64xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>, tensor<1xi32>, tensor<i32>)
      outs(%o, %pk, %pv : tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>
  return %r#0, %r#1, %r#2 : tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>
}

// -----

// A dynamic sequence sizes the attention output from tensor.dim of the
// rotated query. That query reads only the type, so it moves to the rope
// input and does not keep the rope alive.
// CHECK-LABEL: func.func @fold_dynamic_seq
// CHECK-SAME: %[[Q:[^:]*]]: tensor<1x?x64xf16>
// CHECK-NOT: hip.rope
// CHECK: tensor.dim %[[Q]]
// CHECK: hip.gqa
// CHECK-SAME: do_rotary = 1
func.func @fold_dynamic_seq(%ctx: !hip.context, %q: tensor<1x?x64xf16>,
                            %k: tensor<1x?x32xf16>, %v: tensor<1x?x32xf16>,
                            %pk: tensor<1x2x8x16xf16>, %pv: tensor<1x2x8x16xf16>,
                            %sl: tensor<1xi32>, %tot: tensor<i32>,
                            %pos: tensor<1x?xi64>, %cos: tensor<32x8xf16>,
                            %sin: tensor<32x8xf16>)
    -> (tensor<1x?x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>) {
  %c1 = arith.constant 1 : index
  %s = tensor.dim %q, %c1 : tensor<1x?x64xf16>
  %e0 = tensor.empty(%s) : tensor<1x?x64xf16>
  %qr = hip.rope(%ctx) ins(%q, %pos, %cos, %sin : tensor<1x?x64xf16>, tensor<1x?xi64>, tensor<32x8xf16>, tensor<32x8xf16>)
        outs(%e0 : tensor<1x?x64xf16>)
        {interleaved = 0 : i64, num_heads = 4 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x?x64xf16>
  %sk = tensor.dim %k, %c1 : tensor<1x?x32xf16>
  %e1 = tensor.empty(%sk) : tensor<1x?x32xf16>
  %kr = hip.rope(%ctx) ins(%k, %pos, %cos, %sin : tensor<1x?x32xf16>, tensor<1x?xi64>, tensor<32x8xf16>, tensor<32x8xf16>)
        outs(%e1 : tensor<1x?x32xf16>)
        {interleaved = 0 : i64, num_heads = 2 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x?x32xf16>
  %so = tensor.dim %qr, %c1 : tensor<1x?x64xf16>
  %o = tensor.empty(%so) : tensor<1x?x64xf16>
  %r:3 = hip.gqa(%ctx) ins(%qr, %kr, %v, %pk, %pv, %sl, %tot : tensor<1x?x64xf16>, tensor<1x?x32xf16>, tensor<1x?x32xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>, tensor<1xi32>, tensor<i32>)
      outs(%o, %pk, %pv : tensor<1x?x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<1x?x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>
  return %r#0, %r#1, %r#2 : tensor<1x?x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>
}

// -----

// Q and K rotate by different position tensors: hip.gqa has one.
// CHECK-LABEL: func.func @no_fold_different_positions
// CHECK: hip.rope
// CHECK: hip.rope
// CHECK: hip.gqa
// CHECK-NOT: do_rotary = 1
func.func @no_fold_different_positions(%ctx: !hip.context, %q: tensor<1x1x64xf16>,
                       %k: tensor<1x1x32xf16>, %v: tensor<1x1x32xf16>,
                       %pk: tensor<1x2x8x16xf16>, %pv: tensor<1x2x8x16xf16>,
                       %sl: tensor<1xi32>, %tot: tensor<i32>,
                       %pos: tensor<1x1xi64>, %pos2: tensor<1x1xi64>,
                       %cos: tensor<32x8xf16>, %sin: tensor<32x8xf16>)
    -> tensor<1x1x64xf16> {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %qr = hip.rope(%ctx) ins(%q, %pos, %cos, %sin : tensor<1x1x64xf16>, tensor<1x1xi64>, tensor<32x8xf16>, tensor<32x8xf16>)
        outs(%e0 : tensor<1x1x64xf16>)
        {interleaved = 0 : i64, num_heads = 4 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %kr = hip.rope(%ctx) ins(%k, %pos2, %cos, %sin : tensor<1x1x32xf16>, tensor<1x1xi64>, tensor<32x8xf16>, tensor<32x8xf16>)
        outs(%e1 : tensor<1x1x32xf16>)
        {interleaved = 0 : i64, num_heads = 2 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x1x32xf16>
  %o = tensor.empty() : tensor<1x1x64xf16>
  %r:3 = hip.gqa(%ctx) ins(%qr, %kr, %v, %pk, %pv, %sl, %tot : tensor<1x1x64xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>, tensor<1xi32>, tensor<i32>)
      outs(%o, %pk, %pv : tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>
  return %r#0 : tensor<1x1x64xf16>
}

// -----

// Interleaved pairs (2i, 2i+1) are not the half-rotated layout hip.gqa uses.
// CHECK-LABEL: func.func @no_fold_interleaved
// CHECK: hip.rope
// CHECK: hip.rope
// CHECK: hip.gqa
// CHECK-NOT: do_rotary = 1
func.func @no_fold_interleaved(%ctx: !hip.context, %q: tensor<1x1x64xf16>,
                       %k: tensor<1x1x32xf16>, %v: tensor<1x1x32xf16>,
                       %pk: tensor<1x2x8x16xf16>, %pv: tensor<1x2x8x16xf16>,
                       %sl: tensor<1xi32>, %tot: tensor<i32>,
                       %pos: tensor<1x1xi64>, %cos: tensor<32x8xf16>,
                       %sin: tensor<32x8xf16>)
    -> tensor<1x1x64xf16> {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %qr = hip.rope(%ctx) ins(%q, %pos, %cos, %sin : tensor<1x1x64xf16>, tensor<1x1xi64>, tensor<32x8xf16>, tensor<32x8xf16>)
        outs(%e0 : tensor<1x1x64xf16>)
        {interleaved = 1 : i64, num_heads = 4 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %kr = hip.rope(%ctx) ins(%k, %pos, %cos, %sin : tensor<1x1x32xf16>, tensor<1x1xi64>, tensor<32x8xf16>, tensor<32x8xf16>)
        outs(%e1 : tensor<1x1x32xf16>)
        {interleaved = 1 : i64, num_heads = 2 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x1x32xf16>
  %o = tensor.empty() : tensor<1x1x64xf16>
  %r:3 = hip.gqa(%ctx) ins(%qr, %kr, %v, %pk, %pv, %sl, %tot : tensor<1x1x64xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>, tensor<1xi32>, tensor<i32>)
      outs(%o, %pk, %pv : tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>
  return %r#0 : tensor<1x1x64xf16>
}

// -----

// Partial rotary (rotary_embedding_dim < head size) leaves the tail of each
// head unrotated; hip.gqa rotates the whole head.
// CHECK-LABEL: func.func @no_fold_partial_rotary
// CHECK: hip.rope
// CHECK: hip.rope
// CHECK: hip.gqa
// CHECK-NOT: do_rotary = 1
func.func @no_fold_partial_rotary(%ctx: !hip.context, %q: tensor<1x1x64xf16>,
                       %k: tensor<1x1x32xf16>, %v: tensor<1x1x32xf16>,
                       %pk: tensor<1x2x8x16xf16>, %pv: tensor<1x2x8x16xf16>,
                       %sl: tensor<1xi32>, %tot: tensor<i32>,
                       %pos: tensor<1x1xi64>, %cos: tensor<32x4xf16>,
                       %sin: tensor<32x4xf16>)
    -> tensor<1x1x64xf16> {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %qr = hip.rope(%ctx) ins(%q, %pos, %cos, %sin : tensor<1x1x64xf16>, tensor<1x1xi64>, tensor<32x4xf16>, tensor<32x4xf16>)
        outs(%e0 : tensor<1x1x64xf16>)
        {interleaved = 0 : i64, num_heads = 4 : i64, rotary_embedding_dim = 8 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %kr = hip.rope(%ctx) ins(%k, %pos, %cos, %sin : tensor<1x1x32xf16>, tensor<1x1xi64>, tensor<32x4xf16>, tensor<32x4xf16>)
        outs(%e1 : tensor<1x1x32xf16>)
        {interleaved = 0 : i64, num_heads = 2 : i64, rotary_embedding_dim = 8 : i64} : tensor<1x1x32xf16>
  %o = tensor.empty() : tensor<1x1x64xf16>
  %r:3 = hip.gqa(%ctx) ins(%qr, %kr, %v, %pk, %pv, %sl, %tot : tensor<1x1x64xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>, tensor<1xi32>, tensor<i32>)
      outs(%o, %pk, %pv : tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>
  return %r#0 : tensor<1x1x64xf16>
}

// -----

// The rotated query is also a graph output, so the rope has to stay.
// CHECK-LABEL: func.func @no_fold_extra_use
// CHECK: hip.rope
// CHECK: hip.rope
// CHECK: hip.gqa
// CHECK-NOT: do_rotary = 1
func.func @no_fold_extra_use(%ctx: !hip.context, %q: tensor<1x1x64xf16>,
                       %k: tensor<1x1x32xf16>, %v: tensor<1x1x32xf16>,
                       %pk: tensor<1x2x8x16xf16>, %pv: tensor<1x2x8x16xf16>,
                       %sl: tensor<1xi32>, %tot: tensor<i32>,
                       %pos: tensor<1x1xi64>, %cos: tensor<32x8xf16>,
                       %sin: tensor<32x8xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x1x64xf16>) {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %qr = hip.rope(%ctx) ins(%q, %pos, %cos, %sin : tensor<1x1x64xf16>, tensor<1x1xi64>, tensor<32x8xf16>, tensor<32x8xf16>)
        outs(%e0 : tensor<1x1x64xf16>)
        {interleaved = 0 : i64, num_heads = 4 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %kr = hip.rope(%ctx) ins(%k, %pos, %cos, %sin : tensor<1x1x32xf16>, tensor<1x1xi64>, tensor<32x8xf16>, tensor<32x8xf16>)
        outs(%e1 : tensor<1x1x32xf16>)
        {interleaved = 0 : i64, num_heads = 2 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x1x32xf16>
  %o = tensor.empty() : tensor<1x1x64xf16>
  %r:3 = hip.gqa(%ctx) ins(%qr, %kr, %v, %pk, %pv, %sl, %tot : tensor<1x1x64xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>, tensor<1xi32>, tensor<i32>)
      outs(%o, %pk, %pv : tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<1x1x64xf16>, tensor<1x2x8x16xf16>, tensor<1x2x8x16xf16>
  return %r#0, %qr : tensor<1x1x64xf16>, tensor<1x1x64xf16>
}

// -----

// fp32 attention over a quantized cache would run through the cast adapter,
// which cannot follow arbitrary position_ids; only fp16 folds.
// CHECK-LABEL: func.func @no_fold_f32
// CHECK: hip.rope
// CHECK: hip.rope
// CHECK: hip.gqa
// CHECK-NOT: do_rotary = 1
func.func @no_fold_f32(%ctx: !hip.context, %q: tensor<1x1x64xf32>,
                       %k: tensor<1x1x32xf32>, %v: tensor<1x1x32xf32>,
                       %pk: tensor<1x2x8x16xf32>, %pv: tensor<1x2x8x16xf32>,
                       %sl: tensor<1xi32>, %tot: tensor<i32>,
                       %pos: tensor<1x1xi64>, %cos: tensor<32x8xf32>,
                       %sin: tensor<32x8xf32>)
    -> tensor<1x1x64xf32> {
  %e0 = tensor.empty() : tensor<1x1x64xf32>
  %qr = hip.rope(%ctx) ins(%q, %pos, %cos, %sin : tensor<1x1x64xf32>, tensor<1x1xi64>, tensor<32x8xf32>, tensor<32x8xf32>)
        outs(%e0 : tensor<1x1x64xf32>)
        {interleaved = 0 : i64, num_heads = 4 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x1x64xf32>
  %e1 = tensor.empty() : tensor<1x1x32xf32>
  %kr = hip.rope(%ctx) ins(%k, %pos, %cos, %sin : tensor<1x1x32xf32>, tensor<1x1xi64>, tensor<32x8xf32>, tensor<32x8xf32>)
        outs(%e1 : tensor<1x1x32xf32>)
        {interleaved = 0 : i64, num_heads = 2 : i64, rotary_embedding_dim = 16 : i64} : tensor<1x1x32xf32>
  %o = tensor.empty() : tensor<1x1x64xf32>
  %r:3 = hip.gqa(%ctx) ins(%qr, %kr, %v, %pk, %pv, %sl, %tot : tensor<1x1x64xf32>, tensor<1x1x32xf32>, tensor<1x1x32xf32>, tensor<1x2x8x16xf32>, tensor<1x2x8x16xf32>, tensor<1xi32>, tensor<i32>)
      outs(%o, %pk, %pv : tensor<1x1x64xf32>, tensor<1x2x8x16xf32>, tensor<1x2x8x16xf32>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<1x1x64xf32>, tensor<1x2x8x16xf32>, tensor<1x2x8x16xf32>
  return %r#0 : tensor<1x1x64xf32>
}
