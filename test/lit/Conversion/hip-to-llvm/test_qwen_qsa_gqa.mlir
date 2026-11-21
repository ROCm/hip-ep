// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// RUN: hip-mlir-opt %s --assign-op-state-slots --convert-hip-to-llvm | FileCheck %s
// CHECK: llvm.func @wrap_qwen_qsa_group_query_attention
// CHECK: llvm.call @wrap_qwen_qsa_group_query_attention
// CHECK-NOT: "hip.qwen_qsa_gqa"
module {
  func.func @native(%ctx: !hip.context, %q: memref<8x?x6144xf16, 1>,
      %k: memref<8x?x1024xf16, 1>, %hidden: memref<8x?x5120xf16, 1>,
      %kv: memref<8x4x65600x256xf16, 1>, %len: memref<8xi32, 1>,
      %cap: memref<i32, 1>, %w: memref<640x5120xbf16, 1>,
      %n: memref<128xbf16, 1>, %rope: memref<262144x32xf32, 1>,
      %pos: memref<3x8x?xi64, 1>) {
    "hip.qwen_qsa_gqa"(%ctx, %q, %k, %k, %kv, %kv, %len, %cap, %hidden, %w, %n, %n, %n,
        %rope, %rope, %pos, %q, %kv, %kv) :
      (!hip.context, memref<8x?x6144xf16, 1>, memref<8x?x1024xf16, 1>, memref<8x?x1024xf16, 1>,
       memref<8x4x65600x256xf16, 1>, memref<8x4x65600x256xf16, 1>, memref<8xi32, 1>, memref<i32, 1>,
       memref<8x?x5120xf16, 1>, memref<640x5120xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>,
       memref<262144x32xf32, 1>, memref<262144x32xf32, 1>, memref<3x8x?xi64, 1>, memref<8x?x6144xf16, 1>,
       memref<8x4x65600x256xf16, 1>, memref<8x4x65600x256xf16, 1>) -> ()
    return
  }
}
