// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s
//
// The trailing i32 on the wrap_group_query_attention call is op_state_slot,
// threaded by --assign-op-state-slots. It selects this GQA instance's
// per-instance GqaState (hipBLASLt GEMM descriptor cache), replacing the
// former shared RuntimeState::gqa_gemm_cache.


// CHECK-LABEL: llvm.func @test_gqa_lowering
// The rank-3 key operand is BSHD, so the trailing kv_layout_bnsh argument is 0.
// The element_size_bytes constant anchors past no_causal, the one i32 zero
// emitted ahead of it; kv_layout_bnsh is the next i32 constant after it, and
// op_state_slot follows both.
// CHECK: llvm.mlir.constant(2 : i64)
// CHECK: %[[BSHD:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @wrap_group_query_attention({{.*}}, %[[BSHD]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32, i64, i64, f32, i64, i64, i64, i64, i64, i64, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> i32

// Verify 42 parameters (full MS GQA spec signature + no_causal + op_state_slot
// + kv_layout_bnsh):
// - 19 pointers: state, query, key, value, past_key, past_value, seqlens_k, total_seq_len,
//                cos_cache(NULL), sin_cache(NULL), position_ids(NULL), attention_bias(NULL),
//                head_sink(NULL), k_scale(NULL), v_scale(NULL),
//                output, present_key, present_value, output_qk(NULL)
// - 13 attributes: num_heads=32, kv_num_heads=8, scale=0.0883883461, do_rotary=0, rotary_interleaved=0,
//                  softcap=0.0, local_window_size=-1, smooth_softmax=0, qk_output=0,
//                  k_quant_type=0(NONE), v_quant_type=0(NONE), kv_cache_bit_width=8,
//                  no_causal=0(i32)
// - 6 shape params: batch_size=1, seq_len_q=1, seq_len_kv=128, past_buf_seq=127, head_dim=128,
//                   element_size_bytes=2
// - 2 bias broadcast params: attn_bias_batch=1, attn_bias_num_heads=1 (no attention_bias operand)
// - 1 i32: op_state_slot (per-instance GqaState; threaded by
//          --assign-op-state-slots, replaces shared RuntimeState::gqa_gemm_cache)
// - 1 i32: kv_layout_bnsh=0 (rank-3 key operand => BSHD; a rank-4 key would
//          give 1, telling the runtime the source is already in present order)
module {
  func.func @test_gqa_lowering(%ctx: !hip.context,
                                %query: memref<1x1x4096xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x8x127x128xf16, 1>,
                                %past_value: memref<1x8x127x128xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x4096xf16, 1>,
                                %present_key: memref<1x8x128x128xf16, 1>,
                                %present_value: memref<1x8x128x128xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x4096xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x8x127x128xf16, 1>, memref<1x8x127x128xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x4096xf16, 1>, memref<1x8x128x128xf16, 1>, memref<1x8x128x128xf16, 1>)
        {num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0883883461 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// ---
// The rank-4 key operand is BNSH -- already in present order -- so the same
// argument is 1. Anchored the same way as above.
// CHECK-LABEL: llvm.func @test_gqa_cross_attn_bnsh
// CHECK: llvm.mlir.constant(2 : i64)
// CHECK: %[[BNSH:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @wrap_group_query_attention({{.*}}, %[[BNSH]])
module {
  func.func @test_gqa_cross_attn_bnsh(%ctx: !hip.context,
                                %query: memref<1x1x4096xf16, 1>,
                                %key: memref<1x8x1500x128xf16, 1>,
                                %value: memref<1x8x1500x128xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x4096xf16, 1>,
                                %present_key: memref<1x8x1500x128xf16, 1>,
                                %present_value: memref<1x8x1500x128xf16, 1>) {
    "hip.gqa"(%ctx, %query, %key, %value, %seqlens_k, %total_seq_len,
              %output, %present_key, %present_value) <{
        num_heads = 32 : i64, kv_num_heads = 8 : i64,
        scale = 0.0883883461 : f32, softcap = 0.000000e+00 : f32,
        do_rotary = 0 : i64, rotary_interleaved = 0 : i64,
        no_causal = true,
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>
      }> : (!hip.context, memref<1x1x4096xf16, 1>, memref<1x8x1500x128xf16, 1>,
            memref<1x8x1500x128xf16, 1>, memref<1x1xi32, 1>, memref<i32, 1>,
            memref<1x1x4096xf16, 1>, memref<1x8x1500x128xf16, 1>,
            memref<1x8x1500x128xf16, 1>) -> ()
    return
  }
}
