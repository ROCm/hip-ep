// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s
//
// rotary_dim, the argument after rotary_interleaved, is twice the cos_cache
// width: a constant for a static width, read from the memref descriptor for
// a dynamic one. Here head_dim = 256 and cos is [max_pos, 32], so
// rotary_dim = 64 < head_dim (partial rotary).

module {
  func.func @gqa_partial_rotary_static(%ctx: !hip.context,
      %query: memref<1x1x2048xf16, 1>, %key: memref<1x1x512xf16, 1>,
      %value: memref<1x1x512xf16, 1>, %past_key: memref<1x2x7x256xf16, 1>,
      %past_value: memref<1x2x7x256xf16, 1>, %seqlens_k: memref<1x1xi32, 1>,
      %total_seq_len: memref<i32, 1>, %cos: memref<4096x32xf16, 1>,
      %sin: memref<4096x32xf16, 1>, %output: memref<1x1x2048xf16, 1>,
      %present_key: memref<1x2x8x256xf16, 1>,
      %present_value: memref<1x2x8x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k,
            %total_seq_len, %cos, %sin :
            memref<1x1x2048xf16, 1>, memref<1x1x512xf16, 1>,
            memref<1x1x512xf16, 1>, memref<1x2x7x256xf16, 1>,
            memref<1x2x7x256xf16, 1>, memref<1x1xi32, 1>, memref<i32, 1>,
            memref<4096x32xf16, 1>, memref<4096x32xf16, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x2048xf16, 1>, memref<1x2x8x256xf16, 1>,
             memref<1x2x8x256xf16, 1>)
        {num_heads = 8 : i64, kv_num_heads = 2 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 1 : i64, rotary_interleaved = 0 : i64}
    return
  }

  func.func @gqa_partial_rotary_dynamic(%ctx: !hip.context,
      %query: memref<1x1x2048xf16, 1>, %key: memref<1x1x512xf16, 1>,
      %value: memref<1x1x512xf16, 1>, %past_key: memref<1x2x7x256xf16, 1>,
      %past_value: memref<1x2x7x256xf16, 1>, %seqlens_k: memref<1x1xi32, 1>,
      %total_seq_len: memref<i32, 1>, %cos: memref<?x?xf16, 1>,
      %sin: memref<?x?xf16, 1>, %output: memref<1x1x2048xf16, 1>,
      %present_key: memref<1x2x8x256xf16, 1>,
      %present_value: memref<1x2x8x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k,
            %total_seq_len, %cos, %sin :
            memref<1x1x2048xf16, 1>, memref<1x1x512xf16, 1>,
            memref<1x1x512xf16, 1>, memref<1x2x7x256xf16, 1>,
            memref<1x2x7x256xf16, 1>, memref<1x1xi32, 1>, memref<i32, 1>,
            memref<?x?xf16, 1>, memref<?x?xf16, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x2048xf16, 1>, memref<1x2x8x256xf16, 1>,
             memref<1x2x8x256xf16, 1>)
        {num_heads = 8 : i64, kv_num_heads = 2 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 1 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @gqa_partial_rotary_static
// CHECK: %[[W:.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK-NEXT: %[[TWO:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[RD:.*]] = llvm.mul %[[W]], %[[TWO]] : i64
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[RD]], {{.*}})

// CHECK-LABEL: llvm.func @gqa_partial_rotary_dynamic
// CHECK: %[[DW:.*]] = llvm.extractvalue %{{.*}}[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[DTWO:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[DRD:.*]] = llvm.mul %[[DW]], %[[DTWO]] : i64
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[DRD]], {{.*}})
