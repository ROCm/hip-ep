// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s
//
// A cache whose physical last dimension differs from head_dim (here the
// KV-Pareto k8v8_g32_planar_v1 codec: 140 FP16 carriers = 280 bytes per token
// and KV head, versus head_dim 256) lowers to wrap_group_query_attention_v2,
// which appends the four physical cache widths to the host-length ABI.
// A dense cache in the same module keeps wrap_group_query_attention_host_length
// or the Ring attention entry when
// Ring is built (this dense case is a proven-FP16 Ring candidate).

module {
  func.func @packed_width(%ctx: !hip.context,
                          %query: memref<1x1x6144xf16, 1>,
                          %key: memref<1x1x1024xf16, 1>,
                          %value: memref<1x1x1024xf16, 1>,
                          %past_key: memref<1x4x128x140xf16, 1>,
                          %past_value: memref<1x4x128x140xf16, 1>,
                          %seqlens_k: memref<1xi32, 1>,
                          %total_seq_len: memref<i32, 1>,
                          %output: memref<1x1x6144xf16, 1>,
                          %present_key: memref<1x4x128x140xf16, 1>,
                          %present_value: memref<1x4x128x140xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x128x140xf16, 1>, memref<1x4x128x140xf16, 1>,
            memref<1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x140xf16, 1>, memref<1x4x128x140xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }

  func.func @dense_width(%ctx: !hip.context,
                         %query: memref<1x1x6144xf16, 1>,
                         %key: memref<1x1x1024xf16, 1>,
                         %value: memref<1x1x1024xf16, 1>,
                         %past_key: memref<1x4x128x256xf16, 1>,
                         %past_value: memref<1x4x128x256xf16, 1>,
                         %seqlens_k: memref<1xi32, 1>,
                         %total_seq_len: memref<i32, 1>,
                         %output: memref<1x1x6144xf16, 1>,
                         %present_key: memref<1x4x128x256xf16, 1>,
                         %present_value: memref<1x4x128x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>,
            memref<1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
  func.func @packed_host_length(%ctx: !hip.context, %length: i32,
      %query: memref<1x1x6144xf16>, %key: memref<1x1x1024xf16>,
      %value: memref<1x1x1024xf16>, %total: memref<i32>,
      %output: memref<1x1x6144xf16>, %pk: memref<1x4x128x140xf16>,
      %pv: memref<1x4x128x140xf16>) {
    %c0 = arith.constant 0 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %length, %len[%c0] : memref<1xi32>
    hip.gqa(%ctx)
      ins(%query, %key, %value, %pk, %pv, %len, %total :
          memref<1x1x6144xf16>, memref<1x1x1024xf16>, memref<1x1x1024xf16>,
          memref<1x4x128x140xf16>, memref<1x4x128x140xf16>, memref<1xi32>, memref<i32>)
      outs(%output, %pk, %pv :
           memref<1x1x6144xf16>, memref<1x4x128x140xf16>, memref<1x4x128x140xf16>)
      {num_heads = 24 : i64, kv_num_heads = 4 : i64}
    return
  }

}

// CHECK-LABEL: llvm.func @packed_width
// CHECK: %[[W0:.*]] = llvm.mlir.constant(140 : i64) : i64
// CHECK-NEXT: %[[W1:.*]] = llvm.mlir.constant(140 : i64) : i64
// CHECK-NEXT: %[[W2:.*]] = llvm.mlir.constant(140 : i64) : i64
// CHECK-NEXT: %[[W3:.*]] = llvm.mlir.constant(140 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_group_query_attention_v2({{.*}}, %[[W0]], %[[W1]], %[[W2]], %[[W3]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32, i64, i64, f32, i64, i64, i64, i64, i64, i64, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i64, i64, i64, i64) -> i32
// CHECK-NOT: llvm.call @wrap_group_query_attention(

// CHECK-LABEL: llvm.func @dense_width
// CHECK: llvm.call @wrap_group_query_attention{{(_host_length|_ring_f16)}}({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32, i64, i64, f32, i64, i64, i64, i64, i64, i64, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32) -> i32
// CHECK-NOT: wrap_group_query_attention_v2

// CHECK-LABEL: llvm.func @packed_host_length
// CHECK: llvm.store %[[LENGTH:arg[0-9]+]],
// CHECK: %[[PRESENT:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[W0:.*]] = llvm.mlir.constant(140 : i64) : i64
// CHECK-NEXT: %[[W1:.*]] = llvm.mlir.constant(140 : i64) : i64
// CHECK-NEXT: %[[W2:.*]] = llvm.mlir.constant(140 : i64) : i64
// CHECK-NEXT: %[[W3:.*]] = llvm.mlir.constant(140 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_group_query_attention_v2({{.*}}, %[[PRESENT]], %[[LENGTH]], %[[W0]], %[[W1]], %[[W2]], %[[W3]])
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length(
// CHECK-NOT: llvm.call @wrap_group_query_attention_ring_f16(
