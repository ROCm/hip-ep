// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// RUN: hip-mlir-opt %s --assign-op-state-slots --convert-hip-to-llvm | FileCheck %s
// CHECK: llvm.func @wrap_qsa_group_query_attention
// CHECK: llvm.call @wrap_qsa_group_query_attention
// CHECK-NOT: "hip.qsa_gqa"
module {
  func.func @native(%ctx: !hip.context, %q: memref<1x?x4096xf16, 1>,
      %kv: memref<1x32x?x128xf16, 1>, %len: memref<1xi32, 1>, %cap: memref<i32, 1>,
      %w: memref<640x4096xbf16, 1>, %n: memref<128xbf16, 1>,
      %rope: memref<65536x64xf32, 1>) {
    "hip.qsa_gqa"(%ctx, %q, %q, %q, %kv, %kv, %len, %cap, %q, %w, %n, %n,
        %rope, %rope, %q, %kv, %kv) :
      (!hip.context, memref<1x?x4096xf16, 1>, memref<1x?x4096xf16, 1>, memref<1x?x4096xf16, 1>,
       memref<1x32x?x128xf16, 1>, memref<1x32x?x128xf16, 1>, memref<1xi32, 1>, memref<i32, 1>,
       memref<1x?x4096xf16, 1>, memref<640x4096xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>,
       memref<65536x64xf32, 1>, memref<65536x64xf32, 1>, memref<1x?x4096xf16, 1>,
       memref<1x32x?x128xf16, 1>, memref<1x32x?x128xf16, 1>) -> ()
    return
  }
}
