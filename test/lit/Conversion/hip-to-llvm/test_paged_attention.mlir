// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_paged_attention_static(
      %ctx: !hip.context,
      %query: memref<4x128xf16, 1>,
      %key: memref<4x128xf16, 1>,
      %value: memref<4x128xf16, 1>,
      %key_cache: memref<8x16x2x16xf16, 1>,
      %value_cache: memref<8x16x2x16xf16, 1>,
      %cum_seq: memref<2xi32, 1>,
      %past_seqlens: memref<1xi32, 1>,
      %block_table: memref<1x8xi32, 1>,
      %output: memref<4x128xf16, 1>) {
    hip.paged_attention(%ctx)
        ins(%query, %key, %value, %key_cache, %value_cache, %cum_seq,
            %past_seqlens, %block_table :
            memref<4x128xf16, 1>, memref<4x128xf16, 1>, memref<4x128xf16, 1>,
            memref<8x16x2x16xf16, 1>, memref<8x16x2x16xf16, 1>, memref<2xi32, 1>,
            memref<1xi32, 1>, memref<1x8xi32, 1>)
        outs(%output : memref<4x128xf16, 1>)
        {num_heads = 4 : i64, kv_num_heads = 2 : i64, do_rotary = 0 : i64,
         rotary_interleaved = 0 : i64, local_window_size = -1 : i64,
         scale = 0.000000e+00 : f32, softcap = 0.000000e+00 : f32}
    return
  }
}

// CHECK-LABEL: llvm.func @test_paged_attention_static
// CHECK: llvm.call @wrap_paged_attention

// ---

module {
  func.func @test_paged_attention_dynamic_query(
      %ctx: !hip.context,
      %query: memref<?x128xf16, 1>,
      %key: memref<?x128xf16, 1>,
      %value: memref<?x128xf16, 1>,
      %key_cache: memref<8x16x2x16xf16, 1>,
      %value_cache: memref<8x16x2x16xf16, 1>,
      %cum_seq: memref<2xi32, 1>,
      %past_seqlens: memref<1xi32, 1>,
      %block_table: memref<1x8xi32, 1>,
      %output: memref<?x128xf16, 1>) {
    hip.paged_attention(%ctx)
        ins(%query, %key, %value, %key_cache, %value_cache, %cum_seq,
            %past_seqlens, %block_table :
            memref<?x128xf16, 1>, memref<?x128xf16, 1>, memref<?x128xf16, 1>,
            memref<8x16x2x16xf16, 1>, memref<8x16x2x16xf16, 1>,
            memref<2xi32, 1>, memref<1xi32, 1>, memref<1x8xi32, 1>)
        outs(%output : memref<?x128xf16, 1>)
        {num_heads = 4 : i64, kv_num_heads = 2 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @test_paged_attention_dynamic_query
// CHECK: llvm.extractvalue {{.*}}[3, 0]
// CHECK: llvm.call @wrap_paged_attention
