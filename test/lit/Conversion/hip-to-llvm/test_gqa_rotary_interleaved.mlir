// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s
// The runtime RoPE kernels are half-rotated only, so an interleaved rotation
// must fail at compile time rather than return an error on every run. The
// attribute is irrelevant when the op does not rotate. They read the cos/sin
// caches as [max_position, D / 2] rows, so another rank fails here too.

module {
  func.func @interleaved_rotary(%ctx: !hip.context,
                                %query: memref<1x1x4096xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x8x128x128xf16, 1>,
                                %past_value: memref<1x8x128x128xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %cos: memref<256x64xf16, 1>,
                                %sin: memref<256x64xf16, 1>,
                                %output: memref<1x1x4096xf16, 1>) {
    // expected-error @below {{rotary_interleaved = 1 is not supported by the HIP runtime}}
    // expected-error @below {{failed to legalize operation 'hip.gqa'}}
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len, %cos, %sin :
            memref<1x1x4096xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x8x128x128xf16, 1>, memref<1x8x128x128xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>,
            memref<256x64xf16, 1>, memref<256x64xf16, 1>)
        outs(%output, %past_key, %past_value :
             memref<1x1x4096xf16, 1>, memref<1x8x128x128xf16, 1>, memref<1x8x128x128xf16, 1>)
        {num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0883883461 : f32, do_rotary = 1 : i64, rotary_interleaved = 1 : i64}
    return
  }
}

// -----

module {
  func.func @half_rotated(%ctx: !hip.context,
                          %query: memref<1x1x4096xf16, 1>,
                          %key: memref<1x1x1024xf16, 1>,
                          %value: memref<1x1x1024xf16, 1>,
                          %past_key: memref<1x8x128x128xf16, 1>,
                          %past_value: memref<1x8x128x128xf16, 1>,
                          %seqlens_k: memref<1x1xi32, 1>,
                          %total_seq_len: memref<i32, 1>,
                          %cos: memref<256x64xf16, 1>,
                          %sin: memref<256x64xf16, 1>,
                          %output: memref<1x1x4096xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len, %cos, %sin :
            memref<1x1x4096xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x8x128x128xf16, 1>, memref<1x8x128x128xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>,
            memref<256x64xf16, 1>, memref<256x64xf16, 1>)
        outs(%output, %past_key, %past_value :
             memref<1x1x4096xf16, 1>, memref<1x8x128x128xf16, 1>, memref<1x8x128x128xf16, 1>)
        {num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0883883461 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// The cos rows bound every RoPE position, so they travel just before the
// host-length pair.
// CHECK-LABEL: llvm.func @half_rotated
// CHECK: %[[ABSENT:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[ROWS:.*]] = llvm.mlir.constant(256 : i64) : i64
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[ROWS]], %[[ABSENT]], %[[ABSENT]], {{.*}})

// -----

module {
  func.func @rank3_cos(%ctx: !hip.context,
                       %query: memref<1x1x4096xf16, 1>,
                       %key: memref<1x1x1024xf16, 1>,
                       %value: memref<1x1x1024xf16, 1>,
                       %past_key: memref<1x8x128x128xf16, 1>,
                       %past_value: memref<1x8x128x128xf16, 1>,
                       %seqlens_k: memref<1x1xi32, 1>,
                       %total_seq_len: memref<i32, 1>,
                       %cos: memref<1x256x64xf16, 1>,
                       %sin: memref<1x256x64xf16, 1>,
                       %output: memref<1x1x4096xf16, 1>) {
    // expected-error @below {{cos_cache must be rank 2 [max_position, D / 2]}}
    // expected-error @below {{failed to legalize operation 'hip.gqa'}}
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len, %cos, %sin :
            memref<1x1x4096xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x8x128x128xf16, 1>, memref<1x8x128x128xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>,
            memref<1x256x64xf16, 1>, memref<1x256x64xf16, 1>)
        outs(%output, %past_key, %past_value :
             memref<1x1x4096xf16, 1>, memref<1x8x128x128xf16, 1>, memref<1x8x128x128xf16, 1>)
        {num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0883883461 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// -----

module {
  func.func @interleaved_without_rotary(%ctx: !hip.context,
                                        %query: memref<1x1x4096xf16, 1>,
                                        %key: memref<1x1x1024xf16, 1>,
                                        %value: memref<1x1x1024xf16, 1>,
                                        %past_key: memref<1x8x128x128xf16, 1>,
                                        %past_value: memref<1x8x128x128xf16, 1>,
                                        %seqlens_k: memref<1x1xi32, 1>,
                                        %total_seq_len: memref<i32, 1>,
                                        %output: memref<1x1x4096xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x4096xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x8x128x128xf16, 1>, memref<1x8x128x128xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %past_key, %past_value :
             memref<1x1x4096xf16, 1>, memref<1x8x128x128xf16, 1>, memref<1x8x128x128xf16, 1>)
        {num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0883883461 : f32, do_rotary = 0 : i64, rotary_interleaved = 1 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @interleaved_without_rotary
// CHECK: llvm.call @wrap_group_query_attention_host_length(
