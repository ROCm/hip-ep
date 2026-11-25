// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_paged_attention_static(
      %ctx: !hip.context,
      %query: memref<4x64xf16, 1>,
      %key: memref<4x32xf16, 1>,
      %value: memref<4x32xf16, 1>,
      %key_cache: memref<8x16x2x16xf16, 1>,
      %value_cache: memref<8x16x2x16xf16, 1>,
      %cum_seq: memref<2xi32, 1>,
      %past_seqlens: memref<1xi32, 1>,
      %block_table: memref<1x8xi32, 1>,
      %output: memref<4x64xf16, 1>,
      %key_cache_out: memref<8x16x2x16xf16, 1>,
      %value_cache_out: memref<8x16x2x16xf16, 1>) {
    hip.paged_attention(%ctx)
        ins(%query, %key_cache, %value_cache, %cum_seq, %past_seqlens,
            %block_table :
            memref<4x64xf16, 1>, memref<8x16x2x16xf16, 1>,
            memref<8x16x2x16xf16, 1>, memref<2xi32, 1>, memref<1xi32, 1>,
            memref<1x8xi32, 1>)
        key(%key : memref<4x32xf16, 1>)
        value(%value : memref<4x32xf16, 1>)
        outs(%output, %key_cache_out, %value_cache_out :
             memref<4x64xf16, 1>, memref<8x16x2x16xf16, 1>,
             memref<8x16x2x16xf16, 1>)
        {num_heads = 4 : i64, kv_num_heads = 2 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @test_paged_attention_static
// The default scale 1/sqrt(16) is resolved by the compiler.
// CHECK: llvm.mlir.constant(2.500000e-01 : f32)
// CHECK: llvm.call @wrap_paged_attention

// -----

module {
  func.func @test_paged_attention_dynamic_query(
      %ctx: !hip.context,
      %query: memref<?x64xf16, 1>,
      %key: memref<?x32xf16, 1>,
      %value: memref<?x32xf16, 1>,
      %key_cache: memref<8x16x2x16xf16, 1>,
      %value_cache: memref<8x16x2x16xf16, 1>,
      %cum_seq: memref<2xi32, 1>,
      %past_seqlens: memref<1xi32, 1>,
      %block_table: memref<1x8xi32, 1>,
      %output: memref<?x64xf16, 1>,
      %key_cache_out: memref<8x16x2x16xf16, 1>,
      %value_cache_out: memref<8x16x2x16xf16, 1>) {
    hip.paged_attention(%ctx)
        ins(%query, %key_cache, %value_cache, %cum_seq, %past_seqlens,
            %block_table :
            memref<?x64xf16, 1>, memref<8x16x2x16xf16, 1>,
            memref<8x16x2x16xf16, 1>, memref<2xi32, 1>, memref<1xi32, 1>,
            memref<1x8xi32, 1>)
        key(%key : memref<?x32xf16, 1>)
        value(%value : memref<?x32xf16, 1>)
        outs(%output, %key_cache_out, %value_cache_out :
             memref<?x64xf16, 1>, memref<8x16x2x16xf16, 1>, memref<8x16x2x16xf16, 1>)
        {num_heads = 4 : i64, kv_num_heads = 2 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @test_paged_attention_dynamic_query
// CHECK: llvm.extractvalue {{.*}}[3, 0]
// CHECK: llvm.call @wrap_paged_attention

// -----

// Packed QKV: absent key and value reach the runtime as null pointers.
module {
  func.func @test_paged_attention_packed(
      %ctx: !hip.context,
      %query: memref<4x128xf16, 1>,
      %key_cache: memref<8x16x2x16xf16, 1>,
      %value_cache: memref<8x16x2x16xf16, 1>,
      %cum_seq: memref<2xi32, 1>,
      %past_seqlens: memref<1xi32, 1>,
      %block_table: memref<1x8xi32, 1>,
      %output: memref<4x64xf16, 1>,
      %key_cache_out: memref<8x16x2x16xf16, 1>,
      %value_cache_out: memref<8x16x2x16xf16, 1>) {
    hip.paged_attention(%ctx)
        ins(%query, %key_cache, %value_cache, %cum_seq, %past_seqlens,
            %block_table :
            memref<4x128xf16, 1>, memref<8x16x2x16xf16, 1>,
            memref<8x16x2x16xf16, 1>, memref<2xi32, 1>, memref<1xi32, 1>,
            memref<1x8xi32, 1>)
        outs(%output, %key_cache_out, %value_cache_out :
             memref<4x64xf16, 1>, memref<8x16x2x16xf16, 1>, memref<8x16x2x16xf16, 1>)
        {num_heads = 4 : i64, kv_num_heads = 2 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @test_paged_attention_packed
// CHECK: llvm.mlir.zero
// CHECK: llvm.call @wrap_paged_attention

// -----

// Rotary tables, interleaving and a local window are passed through. An
// explicit scale is used as is.
module {
  func.func @test_paged_attention_rotary(
      %ctx: !hip.context,
      %query: memref<4x64xf16, 1>,
      %key: memref<4x32xf16, 1>,
      %value: memref<4x32xf16, 1>,
      %key_cache: memref<8x16x2x16xf16, 1>,
      %value_cache: memref<8x16x2x16xf16, 1>,
      %cum_seq: memref<2xi32, 1>,
      %past_seqlens: memref<1xi32, 1>,
      %block_table: memref<1x8xi32, 1>,
      %cos: memref<32x8xf16, 1>,
      %sin: memref<32x8xf16, 1>,
      %output: memref<4x64xf16, 1>,
      %key_cache_out: memref<8x16x2x16xf16, 1>,
      %value_cache_out: memref<8x16x2x16xf16, 1>) {
    hip.paged_attention(%ctx)
        ins(%query, %key_cache, %value_cache, %cum_seq, %past_seqlens,
            %block_table :
            memref<4x64xf16, 1>, memref<8x16x2x16xf16, 1>,
            memref<8x16x2x16xf16, 1>, memref<2xi32, 1>, memref<1xi32, 1>,
            memref<1x8xi32, 1>)
        key(%key : memref<4x32xf16, 1>)
        value(%value : memref<4x32xf16, 1>)
        cos(%cos : memref<32x8xf16, 1>)
        sin(%sin : memref<32x8xf16, 1>)
        outs(%output, %key_cache_out, %value_cache_out :
             memref<4x64xf16, 1>, memref<8x16x2x16xf16, 1>, memref<8x16x2x16xf16, 1>)
        {num_heads = 4 : i64, kv_num_heads = 2 : i64, do_rotary = 1 : i64,
         rotary_interleaved = 1 : i64, local_window_size = 8 : i64,
         scale = 5.000000e-01 : f32}
    return
  }
}

// CHECK-LABEL: llvm.func @test_paged_attention_rotary
// CHECK: llvm.mlir.constant(5.000000e-01 : f32)
// CHECK: llvm.call @wrap_paged_attention

// -----

// A dynamic head size cannot be folded, so scale 0 is left for the runtime.
module {
  func.func @test_paged_attention_dynamic_head_size(
      %ctx: !hip.context,
      %query: memref<4x64xf16, 1>,
      %key: memref<4x32xf16, 1>,
      %value: memref<4x32xf16, 1>,
      %key_cache: memref<8x16x2x?xf16, 1>,
      %value_cache: memref<8x16x2x?xf16, 1>,
      %cum_seq: memref<2xi32, 1>,
      %past_seqlens: memref<1xi32, 1>,
      %block_table: memref<1x8xi32, 1>,
      %output: memref<4x64xf16, 1>,
      %key_cache_out: memref<8x16x2x?xf16, 1>,
      %value_cache_out: memref<8x16x2x?xf16, 1>) {
    hip.paged_attention(%ctx)
        ins(%query, %key_cache, %value_cache, %cum_seq, %past_seqlens,
            %block_table :
            memref<4x64xf16, 1>, memref<8x16x2x?xf16, 1>,
            memref<8x16x2x?xf16, 1>, memref<2xi32, 1>, memref<1xi32, 1>,
            memref<1x8xi32, 1>)
        key(%key : memref<4x32xf16, 1>)
        value(%value : memref<4x32xf16, 1>)
        outs(%output, %key_cache_out, %value_cache_out :
             memref<4x64xf16, 1>, memref<8x16x2x?xf16, 1>, memref<8x16x2x?xf16, 1>)
        {num_heads = 4 : i64, kv_num_heads = 2 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @test_paged_attention_dynamic_head_size
// CHECK: llvm.mlir.constant(0.000000e+00 : f32)
// CHECK: llvm.call @wrap_paged_attention
