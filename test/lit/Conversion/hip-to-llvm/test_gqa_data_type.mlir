// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --assign-op-state-slots --convert-hip-to-llvm --cse %s | FileCheck %s

// hip.gqa passes the query element type as the last call argument
// (HIPDNN_EP_DATATYPE_*). element_size_bytes is 2 for both fp16 and bf16, so
// without it the runtime would run a bf16 GQA through fp16 GEMMs and softmax.
// --cse leaves one constant per value, so the first match is the operand.

// CHECK-LABEL: llvm.func @gqa_f16
// CHECK: %[[DT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[DT]]) : ({{.*}}, !llvm.ptr, i64, i64, i64) -> i32
func.func @gqa_f16(%ctx: !hip.context, %q: memref<1x4x256xf16, 1>,
    %k: memref<1x4x128xf16, 1>, %v: memref<1x4x128xf16, 1>,
    %past_k: memref<1x2x4x64xf16, 1>, %past_v: memref<1x2x4x64xf16, 1>,
    %seqlens_k: memref<1x1xi32, 1>, %total: memref<i32, 1>,
    %out: memref<1x4x256xf16, 1>,
    %pk: memref<1x2x8x64xf16, 1>, %pv: memref<1x2x8x64xf16, 1>) {
  hip.gqa(%ctx)
      ins(%q, %k, %v, %past_k, %past_v, %seqlens_k, %total :
          memref<1x4x256xf16, 1>, memref<1x4x128xf16, 1>, memref<1x4x128xf16, 1>,
          memref<1x2x4x64xf16, 1>, memref<1x2x4x64xf16, 1>,
          memref<1x1xi32, 1>, memref<i32, 1>)
      outs(%out, %pk, %pv :
           memref<1x4x256xf16, 1>, memref<1x2x8x64xf16, 1>, memref<1x2x8x64xf16, 1>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64, scale = 0.125 : f32,
       softcap = 0.000000e+00 : f32, do_rotary = 0 : i64,
       rotary_interleaved = 0 : i64}
  return
}

// -----

// CHECK-LABEL: llvm.func @gqa_bf16
// CHECK: %[[DT:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[DT]]) : ({{.*}}, !llvm.ptr, i64, i64, i64) -> i32
func.func @gqa_bf16(%ctx: !hip.context, %q: memref<1x4x256xbf16, 1>,
    %k: memref<1x4x128xbf16, 1>, %v: memref<1x4x128xbf16, 1>,
    %past_k: memref<1x2x4x64xbf16, 1>, %past_v: memref<1x2x4x64xbf16, 1>,
    %seqlens_k: memref<1x1xi32, 1>, %total: memref<i32, 1>,
    %out: memref<1x4x256xbf16, 1>,
    %pk: memref<1x2x8x64xbf16, 1>, %pv: memref<1x2x8x64xbf16, 1>) {
  hip.gqa(%ctx)
      ins(%q, %k, %v, %past_k, %past_v, %seqlens_k, %total :
          memref<1x4x256xbf16, 1>, memref<1x4x128xbf16, 1>, memref<1x4x128xbf16, 1>,
          memref<1x2x4x64xbf16, 1>, memref<1x2x4x64xbf16, 1>,
          memref<1x1xi32, 1>, memref<i32, 1>)
      outs(%out, %pk, %pv :
           memref<1x4x256xbf16, 1>, memref<1x2x8x64xbf16, 1>, memref<1x2x8x64xbf16, 1>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64, scale = 0.125 : f32,
       softcap = 0.000000e+00 : f32, do_rotary = 0 : i64,
       rotary_interleaved = 0 : i64}
  return
}

// -----

// CHECK-LABEL: llvm.func @gqa_f32
// CHECK: %[[DT:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[DT]]) : ({{.*}}, !llvm.ptr, i64, i64, i64) -> i32
func.func @gqa_f32(%ctx: !hip.context, %q: memref<1x4x256xf32, 1>,
    %k: memref<1x4x128xf32, 1>, %v: memref<1x4x128xf32, 1>,
    %past_k: memref<1x2x4x64xf32, 1>, %past_v: memref<1x2x4x64xf32, 1>,
    %seqlens_k: memref<1x1xi32, 1>, %total: memref<i32, 1>,
    %out: memref<1x4x256xf32, 1>,
    %pk: memref<1x2x8x64xf32, 1>, %pv: memref<1x2x8x64xf32, 1>) {
  hip.gqa(%ctx)
      ins(%q, %k, %v, %past_k, %past_v, %seqlens_k, %total :
          memref<1x4x256xf32, 1>, memref<1x4x128xf32, 1>, memref<1x4x128xf32, 1>,
          memref<1x2x4x64xf32, 1>, memref<1x2x4x64xf32, 1>,
          memref<1x1xi32, 1>, memref<i32, 1>)
      outs(%out, %pk, %pv :
           memref<1x4x256xf32, 1>, memref<1x2x8x64xf32, 1>, memref<1x2x8x64xf32, 1>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64, scale = 0.125 : f32,
       softcap = 0.000000e+00 : f32, do_rotary = 0 : i64,
       rotary_interleaved = 0 : i64}
  return
}
