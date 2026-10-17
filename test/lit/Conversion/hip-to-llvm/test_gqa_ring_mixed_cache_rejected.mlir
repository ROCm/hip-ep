// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// REQUIRES: ring-gqa
// RUN: not hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s 2>&1 | FileCheck %s
//
// The trailing i32 on the wrap_group_query_attention call is op_state_slot,
// threaded by --assign-op-state-slots. It selects this GQA instance's
// per-instance GqaState (hipBLASLt GEMM descriptor cache), replacing the
// former shared RuntimeState::gqa_gemm_cache.

module {
  func.func @test_gqa_lowering(%ctx: !hip.context,
                                %query: memref<1x1x4096xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x8x127x128xbf16, 1>,
                                %past_value: memref<1x8x127x128xbf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x4096xf16, 1>,
                                %present_key: memref<1x8x128x128xbf16, 1>,
                                %present_value: memref<1x8x128x128xbf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x4096xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x8x127x128xbf16, 1>, memref<1x8x127x128xbf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x4096xf16, 1>, memref<1x8x128x128xbf16, 1>, memref<1x8x128x128xbf16, 1>)
        {num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0883883461 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK: Ring GQA builds require matching unquantized cache types
