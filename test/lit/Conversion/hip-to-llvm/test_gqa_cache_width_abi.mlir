// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s
// Physical-width ABI carries compiler-proven FP16 types even when Ring is off.

// -----
module {
  func.func @packed_f16(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x66xf16, 1>,
                                %past_value: memref<1x4x127x66xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x66xf16, 1>,
                                %present_value: memref<1x4x128x66xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x66xf16, 1>, memref<1x4x127x66xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x66xf16, 1>, memref<1x4x128x66xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @packed_f16(
// CHECK: %[[WIDTH:.*]] = llvm.mlir.constant(66 : i64) : i64
// CHECK: %[[FP16:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @wrap_group_query_attention_v4({{.*}}, %[[FP16]])

// -----
module {
  func.func @dynamic_f16(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x?xf16, 1>,
                                %past_value: memref<1x4x127x?xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x?xf16, 1>,
                                %present_value: memref<1x4x128x?xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x?xf16, 1>, memref<1x4x127x?xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x?xf16, 1>, memref<1x4x128x?xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @dynamic_f16(
// CHECK: %[[FP16:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @wrap_group_query_attention_v4({{.*}}, %[[FP16]])

// -----
module {
  func.func @packed_bf16(%ctx: !hip.context,
                                %query: memref<1x1x6144xbf16, 1>,
                                %key: memref<1x1x1024xbf16, 1>,
                                %value: memref<1x1x1024xbf16, 1>,
                                %past_key: memref<1x4x127x66xbf16, 1>,
                                %past_value: memref<1x4x127x66xbf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xbf16, 1>,
                                %present_key: memref<1x4x128x66xbf16, 1>,
                                %present_value: memref<1x4x128x66xbf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xbf16, 1>, memref<1x1x1024xbf16, 1>, memref<1x1x1024xbf16, 1>,
            memref<1x4x127x66xbf16, 1>, memref<1x4x127x66xbf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xbf16, 1>, memref<1x4x128x66xbf16, 1>, memref<1x4x128x66xbf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @packed_bf16(
// CHECK: %[[WIDTH:.*]] = llvm.mlir.constant(66 : i64) : i64
// CHECK: %[[FP16:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @wrap_group_query_attention_v4({{.*}}, %[[FP16]])

// -----
module {
  func.func @packed_mixed(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x66xf16, 1>,
                                %past_value: memref<1x4x127x66xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x66xbf16, 1>,
                                %present_value: memref<1x4x128x66xbf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x66xf16, 1>, memref<1x4x127x66xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x66xbf16, 1>, memref<1x4x128x66xbf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @packed_mixed(
// CHECK: %[[WIDTH:.*]] = llvm.mlir.constant(66 : i64) : i64
// CHECK: %[[FP16:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @wrap_group_query_attention_v4({{.*}}, %[[FP16]])
