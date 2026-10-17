// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.linear_attention lowers to a single call to
// wrap_linear_attention with 28 parameters:
// - 12 pointers: state, query, key, value, past_state, decay, beta,
//                decay_bias, decay_scale, qk_norm_weight, output,
//                present_state (absent optionals become null)
// - 6 attributes: q_num_heads, kv_num_heads, n_k_heads, scale, chunk_size,
//                 update_rule  (n_k_heads is derived at runtime as
//                 key.dim[2] / head_dim_k, emitted via a second llvm.udiv)
// - 2 optional-input layout flags: decay_per_key_dim, beta_per_head
//                 (i64; 0 when the corresponding operand is absent, otherwise
//                 derived from the operand's last dim at lowering time:
//                 decay_per_key_dim = 1 if decay.dim[-1] != H_kv,
//                 beta_per_head     = 1 if beta.dim[-1]  != 1)
// - 5 shape params: batch_size, seq_len, head_dim_k, head_dim_v,
//                   type / HIPDNN_EP_DATATYPE_* (static -> llvm.mlir.constant,
//                   dynamic -> llvm.extractvalue %desc[3, N], with
//                   head_dim_k = query.dim[2] / q_num_heads always via
//                   llvm.udiv)
// - 3 input-preparation params: qk_norm_epsilon, q_scale, beta_sigmoid
//
// Test cases:
// 1. test_linear_attention_lowering          - all static, gated + past/decay
// 2. test_linear_attention_dynamic_prefill   - dynamic batch + seq_len,
//                                               hidden dims static (typical
//                                               LLM prefill, no past_state)
// 3. test_linear_attention_fully_dynamic     - all tensor dims dynamic;
//                                               validates that head_dim_k and
//                                               head_dim_v fall back to
//                                               runtime extraction rather
//                                               than the kDynamic sentinel.
// 4. test_linear_attention_prepared          - packed QKV with qk norm,
//                                               q scale, softplus decay and
//                                               sigmoid beta.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_linear_attention_lowering(
      %ctx: !hip.context,
      %query: memref<1x1x4096xf16, 1>,
      %key: memref<1x1x1024xf16, 1>,
      %value: memref<1x1x1024xf16, 1>,
      %past_state: memref<1x8x128x128xf16, 1>,
      %decay: memref<1x1x1024xf16, 1>,
      %output: memref<1x1x4096xf16, 1>,
      %present_state: memref<1x8x128x128xf16, 1>) {
    hip.linear_attention(%ctx)
        ins(%query, %key, %value :
            memref<1x1x4096xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x1x1024xf16, 1>)
        past_state(%past_state : memref<1x8x128x128xf16, 1>)
        decay(%decay : memref<1x1x1024xf16, 1>)
        outs(%output, %present_state :
             memref<1x1x4096xf16, 1>, memref<1x8x128x128xf16, 1>)
        {q_num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0883883461 : f32, update_rule = "gated"}

    // Test 1 verifies 28 parameters:
    // - 12 pointers: state, query, key, value, past_state, decay(ptr),
    //                beta(NULL), decay_bias(NULL), decay_scale(NULL),
    //                qk_norm_weight(NULL), output, present_state
    // - 6 attributes: q_num_heads=32, kv_num_heads=8, n_k_heads=8 (=1024/128),
    //                 scale=0.0883883461, chunk_size=0, update_rule=1(gated)
    // - 2 layout flags: decay_per_key_dim=1 (decay dim[-1]=1024 != H_kv=8),
    //                   beta_per_head=0 (beta absent)
    // - 5 shape params: batch_size=1, seq_len=1, head_dim_k=128,
    //                   head_dim_v=128, type=1 (HIPDNN_EP_DATATYPE_HALF for f16)
    // CHECK-LABEL: llvm.func @test_linear_attention_lowering
    // CHECK: llvm.call @wrap_linear_attention({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, f32, i64, i64, i64, i64, i64, i64, i64, f32, f32, i64) -> i32
    return
  }

  // --------------------------------------------------------------------------
  // Test 2: Dynamic batch + seq_len, static hidden dims (prefill w/o state).
  //
  // Expectations:
  // - batch_size  = llvm.extractvalue %query_desc[3, 0]
  // - seq_len     = llvm.extractvalue %query_desc[3, 1]
  // - head_dim_k  = llvm.udiv(const 4096, const q_num_heads=32)
  //                 (UDivOp is always emitted; both operands are constants
  //                  here since the query hidden dim is static)
  // - head_dim_v  = llvm.mlir.constant(128 : i64) from present_state dim 3
  // --------------------------------------------------------------------------
  func.func @test_linear_attention_dynamic_prefill(
      %ctx: !hip.context,
      %query: memref<?x?x4096xf16, 1>,
      %key: memref<?x?x1024xf16, 1>,
      %value: memref<?x?x1024xf16, 1>,
      %output: memref<?x?x4096xf16, 1>,
      %present_state: memref<?x8x128x128xf16, 1>) {
    // CHECK-LABEL: llvm.func @test_linear_attention_dynamic_prefill

    hip.linear_attention(%ctx)
        ins(%query, %key, %value :
            memref<?x?x4096xf16, 1>, memref<?x?x1024xf16, 1>,
            memref<?x?x1024xf16, 1>)
        outs(%output, %present_state :
             memref<?x?x4096xf16, 1>, memref<?x8x128x128xf16, 1>)
        {q_num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0883883461 : f32, update_rule = "linear"}

    // Dynamic batch + seq_len extracted from query descriptor.
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // head_dim_k: query hidden is static (4096), so both udiv operands are
    // constants.
    // CHECK: llvm.mlir.constant(4096 : i64)
    // CHECK: llvm.udiv
    // n_k_heads: key hidden is static (1024), second udiv divides by d_k.
    // CHECK: llvm.mlir.constant(1024 : i64)
    // CHECK: llvm.udiv
    // head_dim_v: present_state dim 3 is static.
    // CHECK: llvm.mlir.constant(128 : i64)
    // CHECK: llvm.call @wrap_linear_attention({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, f32, i64, i64, i64, i64, i64, i64, i64, f32, f32, i64) -> i32

    return
  }

  // --------------------------------------------------------------------------
  // Test 3: Fully dynamic (query and present_state all dynamic).
  //
  // Regression case for the kDynamic-sentinel bug: without dynamic-shape
  // support, head_dim_k would fold to kDynamic (a large negative constant)
  // and head_dim_v would read the static dim (also kDynamic).
  //
  // Expectations:
  // - batch_size, seq_len, query_hidden all extracted from query descriptor
  //   at indices [3, 0..2].
  // - head_dim_k emitted via llvm.udiv with the extracted hidden dim.
  // - head_dim_v extracted from present_state descriptor at index [3, 3].
  // --------------------------------------------------------------------------
  func.func @test_linear_attention_fully_dynamic(
      %ctx: !hip.context,
      %query: memref<?x?x?xf16, 1>,
      %key: memref<?x?x?xf16, 1>,
      %value: memref<?x?x?xf16, 1>,
      %past_state: memref<?x?x?x?xf16, 1>,
      %output: memref<?x?x?xf16, 1>,
      %present_state: memref<?x?x?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @test_linear_attention_fully_dynamic

    hip.linear_attention(%ctx)
        ins(%query, %key, %value :
            memref<?x?x?xf16, 1>, memref<?x?x?xf16, 1>,
            memref<?x?x?xf16, 1>)
        past_state(%past_state : memref<?x?x?x?xf16, 1>)
        outs(%output, %present_state :
             memref<?x?x?xf16, 1>, memref<?x?x?x?xf16, 1>)
        {q_num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0883883461 : f32, update_rule = "gated_delta"}

    // All 3 query dims extracted at runtime.
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // CHECK: llvm.extractvalue {{.*}}[3, 2]
    // head_dim_k = query_hidden / q_num_heads at runtime.
    // CHECK: llvm.udiv
    // n_k_heads = key_hidden / head_dim_k; key_hidden extracted at runtime.
    // CHECK: llvm.extractvalue {{.*}}[3, 2]
    // CHECK: llvm.udiv
    // head_dim_v extracted from present_state descriptor (dim 3).
    // CHECK: llvm.extractvalue {{.*}}[3, 3]
    // CHECK: llvm.call @wrap_linear_attention({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, f32, i64, i64, i64, i64, i64, i64, i64, f32, f32, i64) -> i32

    return
  }

  // --------------------------------------------------------------------------
  // Test 4: Prepared inputs (Qwen3.5 gated-delta-net decode shape).
  //
  // Expectations:
  // - key / value pointers are null (packed QKV).
  // - head_dim_k comes from present_state dim 2, n_k from
  //   (8192 - 16*128 - 32*128) / 128.
  // - decay_bias / decay_scale / qk_norm_weight pointers are passed.
  // - qk_norm_epsilon, q_scale, beta_sigmoid = 1.
  // --------------------------------------------------------------------------
  func.func @test_linear_attention_prepared(
      %ctx: !hip.context,
      %qkv: memref<1x1x8192xf16, 1>,
      %past_state: memref<1x32x128x128xf16, 1>,
      %a: memref<1x1x32xf16, 1>,
      %b: memref<1x1x32xf16, 1>,
      %dt_bias: memref<32xf32, 1>,
      %neg_exp_a: memref<32xf32, 1>,
      %w: memref<128xf16, 1>,
      %output: memref<1x1x4096xf16, 1>,
      %present_state: memref<1x32x128x128xf16, 1>) {
    // CHECK-LABEL: llvm.func @test_linear_attention_prepared

    hip.linear_attention(%ctx)
        ins(%qkv : memref<1x1x8192xf16, 1>)
        past_state(%past_state : memref<1x32x128x128xf16, 1>)
        decay(%a : memref<1x1x32xf16, 1>)
        beta(%b : memref<1x1x32xf16, 1>)
        decay_softplus(%dt_bias, %neg_exp_a :
                       memref<32xf32, 1>, memref<32xf32, 1>)
        qk_norm(%w : memref<128xf16, 1>)
        outs(%output, %present_state :
             memref<1x1x4096xf16, 1>, memref<1x32x128x128xf16, 1>)
        {q_num_heads = 16 : i64, kv_num_heads = 32 : i64, scale = 1.0 : f32,
         chunk_size = 64 : i64, update_rule = "gated_delta",
         qk_norm_epsilon = 1.0e-06 : f32, q_scale = 0.0883789062 : f32,
         beta_sigmoid}

    // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[HQ:.*]] = llvm.mlir.constant(16 : i64)
    // CHECK: %[[HKV:.*]] = llvm.mlir.constant(32 : i64)
    // CHECK: %[[W:.*]] = llvm.mlir.constant(8192 : i64)
    // CHECK: %[[DV:.*]] = llvm.mlir.constant(128 : i64)
    // CHECK: %[[DK:.*]] = llvm.mlir.constant(128 : i64)
    // CHECK: %[[QW:.*]] = llvm.mul %[[HQ]], %[[DK]]
    // CHECK: %[[VW:.*]] = llvm.mul %[[HKV]], %[[DV]]
    // CHECK: %[[R:.*]] = llvm.sub %[[W]], %[[QW]]
    // CHECK: %[[KW:.*]] = llvm.sub %[[R]], %[[VW]]
    // CHECK: %[[NK:.*]] = llvm.udiv %[[KW]], %[[DK]]
    // CHECK: %[[EPS:.*]] = llvm.mlir.constant(9.99999997E-7 : f32)
    // CHECK: %[[QS:.*]] = llvm.mlir.constant(0.0883789062 : f32)
    // CHECK: %[[SIG:.*]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @wrap_linear_attention(%{{[^,]*}}, %{{[^,]*}}, %[[NULL]], %[[NULL]], %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[HQ]], %[[HKV]], %[[NK]], {{.*}}, %[[DK]], %[[DV]], %{{[^,]*}}, %[[EPS]], %[[QS]], %[[SIG]])

    return
  }
}
