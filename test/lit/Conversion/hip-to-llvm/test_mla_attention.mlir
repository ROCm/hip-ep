// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.mla_attention lowers to a wrap_mla_attention call:
// - 8 pointers (query, cache, k_b, v_b, k_b_scales, v_b_scales, kv_seqlen,
//   output), null for absent optional operands
// - the five geometry attributes, scale and block_size as constants
// - batch and query length from query dims 0/1, cache length from cache dim 2
// ============================================================================

// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s

module {
  // ===== int4 up-projections, valid-row count, dynamic lengths =====
  func.func @mla_int4(%ctx: !hip.context,
                      %q: memref<1x?x4096xf16, 1>,
                      %cache: memref<1x1x?x320xf16, 1>,
                      %kb: memref<2048x2x64xui8, 1>,
                      %vb: memref<4096x2x64xui8, 1>,
                      %ks: memref<2048x2xf16, 1>,
                      %vs: memref<4096x2xf16, 1>,
                      %lengths: memref<1xi64, 1>,
                      %out: memref<1x?x4096xf16, 1>) {
    hip.mla_attention(%ctx)
        ins(%q, %cache, %kb, %vb : memref<1x?x4096xf16, 1>,
            memref<1x1x?x320xf16, 1>, memref<2048x2x64xui8, 1>,
            memref<4096x2x64xui8, 1>)
        k_b_scales(%ks : memref<2048x2xf16, 1>)
        v_b_scales(%vs : memref<4096x2xf16, 1>)
        kv_seqlen(%lengths : memref<1xi64, 1>)
        outs(%out : memref<1x?x4096xf16, 1>)
        {num_heads = 32 : i64, kv_lora_rank = 256 : i64, qk_nope_dim = 64 : i64,
         qk_rope_dim = 64 : i64, v_dim = 128 : i64, scale = 0.195 : f32,
         block_size = 128 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @mla_int4
  // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[H:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK-NEXT: %[[L:.*]] = llvm.mlir.constant(256 : i64) : i64
  // CHECK-NEXT: %[[DN:.*]] = llvm.mlir.constant(64 : i64) : i64
  // CHECK-NEXT: %[[R:.*]] = llvm.mlir.constant(64 : i64) : i64
  // CHECK-NEXT: %[[DV:.*]] = llvm.mlir.constant(128 : i64) : i64
  // CHECK-NEXT: %[[SCALE:.*]] = llvm.mlir.constant(1.950000e-01 : f32) : f32
  // CHECK-NEXT: %[[BS:.*]] = llvm.mlir.constant(128 : i64) : i64
  // CHECK-NEXT: llvm.call @wrap_mla_attention(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[H]], %[[L]], %[[DN]], %[[R]], %[[DV]], %[[SCALE]], %[[BS]], %{{[^,]*}}, %{{[^,]*}}, %{{[^,)]*}})
  // CHECK-SAME: (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, f32, i64, i64, i64, i64) -> i32

  // ===== fp16 up-projections: no scales, no valid-row count =====
  func.func @mla_fp16(%ctx: !hip.context,
                      %q: memref<2x1x4096xf16, 1>,
                      %cache: memref<2x1x77x320xf16, 1>,
                      %kb: memref<256x2048xf16, 1>,
                      %vb: memref<256x4096xf16, 1>,
                      %out: memref<2x1x4096xf16, 1>) {
    hip.mla_attention(%ctx)
        ins(%q, %cache, %kb, %vb : memref<2x1x4096xf16, 1>,
            memref<2x1x77x320xf16, 1>, memref<256x2048xf16, 1>,
            memref<256x4096xf16, 1>)
        outs(%out : memref<2x1x4096xf16, 1>)
        {num_heads = 32 : i64, kv_lora_rank = 256 : i64, qk_nope_dim = 64 : i64,
         qk_rope_dim = 64 : i64, v_dim = 128 : i64, scale = 0.195 : f32}
    return
  }

  // CHECK-LABEL: llvm.func @mla_fp16
  // CHECK: %[[NULL2:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK-NEXT: %[[B:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK-NEXT: %[[S:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: %[[T:.*]] = llvm.mlir.constant(77 : i64) : i64
  // CHECK: %[[BS0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: llvm.call @wrap_mla_attention(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[NULL2]], %[[NULL2]], %[[NULL2]], %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[BS0]], %[[B]], %[[S]], %[[T]])
}
