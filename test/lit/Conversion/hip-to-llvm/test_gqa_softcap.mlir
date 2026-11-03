// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s
// Nonzero softcap retains upstream GQA lowering in both Ring ON and OFF builds.
// This verifies routing compatibility, not upstream softcap implementation.

module {
  func.func @softcap_f16(%ctx: !hip.context,
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
         scale = 0.0883883461 : f32, softcap = 2.000000e+01 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @softcap_f16
// CHECK: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @softcap_f32(%ctx: !hip.context,
                                %query: memref<1x1x4096xf32, 1>,
                                %key: memref<1x1x1024xf32, 1>,
                                %value: memref<1x1x1024xf32, 1>,
                                %past_key: memref<1x8x127x128xf32, 1>,
                                %past_value: memref<1x8x127x128xf32, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x4096xf32, 1>,
                                %present_key: memref<1x8x128x128xf32, 1>,
                                %present_value: memref<1x8x128x128xf32, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x4096xf32, 1>, memref<1x1x1024xf32, 1>, memref<1x1x1024xf32, 1>,
            memref<1x8x127x128xf32, 1>, memref<1x8x127x128xf32, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x4096xf32, 1>, memref<1x8x128x128xf32, 1>, memref<1x8x128x128xf32, 1>)
        {num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0883883461 : f32, softcap = -2.000000e+01 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @softcap_f32
// CHECK: llvm.call @wrap_group_query_attention(
