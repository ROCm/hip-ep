// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics

module {
  func.func @strided_pool(%ctx: !hip.context,
      %q: memref<1x256xf16>, %k: memref<1x128xf16>, %v: memref<1x128xf16>,
      %cum: memref<2xi32>, %past: memref<1xi32>, %table: memref<1x2xi32>,
      %out: memref<1x256xf16>,
      %kp: memref<2x16x2x64xf16, strided<[4096, 256, 128, 1]>>,
      %vp: memref<2x16x2x64xf16>) {
    // expected-error @+1 {{requires contiguous row-major memrefs}}
    hip.paged_attention(%ctx) ins(%q, %k, %v, %cum, %past, %table :
        memref<1x256xf16>, memref<1x128xf16>, memref<1x128xf16>,
        memref<2xi32>, memref<1xi32>, memref<1x2xi32>)
      outs(%out, %kp, %vp : memref<1x256xf16>,
        memref<2x16x2x64xf16, strided<[4096, 256, 128, 1]>>,
        memref<2x16x2x64xf16>) {num_heads = 4 : i64, kv_num_heads = 2 : i64}
    return
  }
}

// -----

module {
  func.func @wrong_k_width(%ctx: !hip.context,
      %q: memref<1x256xf16>, %k: memref<1x256xf16>, %v: memref<1x128xf16>,
      %cum: memref<2xi32>, %past: memref<1xi32>, %table: memref<1x2xi32>,
      %out: memref<1x256xf16>, %kp: memref<2x16x2x64xf16>,
      %vp: memref<2x16x2x64xf16>) {
    // expected-error @+1 {{inconsistent Q/K/V/cache/request shapes or element types}}
    hip.paged_attention(%ctx) ins(%q, %k, %v, %cum, %past, %table :
        memref<1x256xf16>, memref<1x256xf16>, memref<1x128xf16>,
        memref<2xi32>, memref<1xi32>, memref<1x2xi32>)
      outs(%out, %kp, %vp : memref<1x256xf16>, memref<2x16x2x64xf16>,
        memref<2x16x2x64xf16>) {num_heads = 4 : i64, kv_num_heads = 2 : i64}
    return
  }
}
