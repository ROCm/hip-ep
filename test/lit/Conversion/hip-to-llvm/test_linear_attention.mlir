// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

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
    return
  }
}

// CHECK-LABEL: llvm.func @test_linear_attention_lowering
// CHECK: llvm.call @wrap_linear_attention({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32, i64, i64, i64, i64, i64, i64, i64) -> i32

// Verify 19 parameters:
// - 9 pointers: state, query, key, value, past_state, decay(ptr), beta(NULL),
//               output, present_state
// - 5 attributes: q_num_heads=32, kv_num_heads=8, scale=0.0883883461,
//                 chunk_size=0, update_rule=1(gated)
// - 5 shape params: batch_size=1, seq_len=1, head_dim_k=128,
//                   head_dim_v=128, element_size_bytes=2
