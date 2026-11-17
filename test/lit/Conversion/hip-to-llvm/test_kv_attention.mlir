// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.kv_attention lowers to a wrap_kv_attention call:
// - 5 pointers (query, key, value, kv_seqlen, output), null without kv_seqlen
// - num_heads, kv_num_heads and scale as constants
// - batch and query length from query dims 0/1, buffer rows from key dim 2,
//   then the key and value head dims
// - key/value head dims that differ, or non-fp16 tensors, are compile errors
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm --split-input-file --verify-diagnostics %s | FileCheck %s

module {
  // ===== GQA heads, valid-row count, dynamic lengths =====
  func.func @kv_lengths(%ctx: !hip.context,
                        %q: memref<1x?x1024xf16, 1>,
                        %k: memref<1x2x?x128xf16, 1>,
                        %v: memref<1x2x?x128xf16, 1>,
                        %lengths: memref<1xi64, 1>,
                        %out: memref<1x?x1024xf16, 1>) {
    hip.kv_attention(%ctx)
        ins(%q, %k, %v : memref<1x?x1024xf16, 1>, memref<1x2x?x128xf16, 1>,
            memref<1x2x?x128xf16, 1>)
        kv_seqlen(%lengths : memref<1xi64, 1>)
        outs(%out : memref<1x?x1024xf16, 1>)
        {num_heads = 8 : i64, kv_num_heads = 2 : i64, scale = 0.0625 : f32}
    return
  }

  // CHECK-LABEL: llvm.func @kv_lengths
  // CHECK: %[[H:.*]] = llvm.mlir.constant(8 : i64) : i64
  // CHECK-NEXT: %[[G:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK-NEXT: %[[SCALE:.*]] = llvm.mlir.constant(6.250000e-02 : f32) : f32
  // CHECK: %[[DK:.*]] = llvm.mlir.constant(128 : i64) : i64
  // CHECK-NEXT: %[[DV:.*]] = llvm.mlir.constant(128 : i64) : i64
  // CHECK-NEXT: llvm.call @wrap_kv_attention(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[H]], %[[G]], %[[SCALE]], %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[DK]], %[[DV]])
  // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32, i64, i64, i64, i64, i64) -> i32

  // ===== no kv_seqlen: null pointer, static geometry =====
  func.func @all_rows(%ctx: !hip.context,
                      %q: memref<2x1x512xf16, 1>,
                      %k: memref<2x8x77x64xf16, 1>,
                      %v: memref<2x8x77x64xf16, 1>,
                      %out: memref<2x1x512xf16, 1>) {
    hip.kv_attention(%ctx)
        ins(%q, %k, %v : memref<2x1x512xf16, 1>, memref<2x8x77x64xf16, 1>,
            memref<2x8x77x64xf16, 1>)
        outs(%out : memref<2x1x512xf16, 1>)
        {num_heads = 8 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32}
    return
  }

  // CHECK-LABEL: llvm.func @all_rows
  // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[B:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK-NEXT: %[[S:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: %[[T:.*]] = llvm.mlir.constant(77 : i64) : i64
  // CHECK: llvm.call @wrap_kv_attention(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[NULL]], %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[B]], %[[S]], %[[T]], %{{[^,]*}}, %{{[^,)]*}})
}

// -----

func.func @split_head_dims(%ctx: !hip.context, %q: memref<1x1x1536xf16, 1>,
    %k: memref<1x8x16x192xf16, 1>, %v: memref<1x8x16x128xf16, 1>,
    %out: memref<1x1x1024xf16, 1>) {
  // expected-error @+2 {{runtime needs one head dim for key and value}}
  // expected-error @+1 {{failed to legalize operation 'hip.kv_attention'}}
  hip.kv_attention(%ctx)
      ins(%q, %k, %v : memref<1x1x1536xf16, 1>, memref<1x8x16x192xf16, 1>,
          memref<1x8x16x128xf16, 1>)
      outs(%out : memref<1x1x1024xf16, 1>)
      {num_heads = 8 : i64, kv_num_heads = 8 : i64, scale = 0.072 : f32}
  return
}

// -----

func.func @f32(%ctx: !hip.context, %q: memref<1x1x512xf32, 1>,
    %k: memref<1x8x16x64xf32, 1>, %v: memref<1x8x16x64xf32, 1>,
    %out: memref<1x1x512xf32, 1>) {
  // expected-error @+2 {{runtime supports fp16 only}}
  // expected-error @+1 {{failed to legalize operation 'hip.kv_attention'}}
  hip.kv_attention(%ctx)
      ins(%q, %k, %v : memref<1x1x512xf32, 1>, memref<1x8x16x64xf32, 1>,
          memref<1x8x16x64xf32, 1>)
      outs(%out : memref<1x1x512xf32, 1>)
      {num_heads = 8 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32}
  return
}
