// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s
// CHECK: llvm.func @wrap_paged_attention_mrope(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32) -> i32
// CHECK-LABEL: llvm.func @paged_mrope_extents
// CHECK: llvm.call @wrap_paged_attention_mrope
module {
  func.func @paged_mrope_extents(
      %ctx: !hip.context,
      %co: memref<8192x32xf32, 1>, %si: memref<8192x32xf32, 1>, %pos: memref<3x?xi64, 1>,
      %q: memref<?x6144xf32, 1>,
      %k: memref<?x1024xf32, 1>,
      %v: memref<?x1024xf16, 1>,
      %kp: memref<?x?x4x256xf16, 1>,
      %vp: memref<?x?x4x256xf16, 1>,
      %cum: memref<?xi32, 1>,
      %past: memref<?xi32, 1>,
      %table: memref<?x?xi32, 1>,
      %out: memref<?x6144xf16, 1>) {
    hip.paged_mrope_attention(%ctx)
        ins(%q, %k, %v, %cum, %past, %table :
            memref<?x6144xf32, 1>, memref<?x1024xf32, 1>,
            memref<?x1024xf16, 1>, memref<?xi32, 1>, memref<?xi32, 1>,
            memref<?x?xi32, 1>)
        rotary(%co, %si, %pos : memref<8192x32xf32, 1>, memref<8192x32xf32, 1>, memref<3x?xi64, 1>)
        outs(%out, %kp, %vp : memref<?x6144xf16, 1>,
            memref<?x?x4x256xf16, 1>, memref<?x?x4x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         rotary_dim = 64 : i64, mrope_section = [11, 11, 10], mrope_layout = 1 : i64, hip.op_state_slot = 0 : i32}
    return
  }
}
