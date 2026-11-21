// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics
module {
  func.func @valid(%ctx: !hip.context, %q: memref<8x?x6144xf16, 1>,
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

// -----
module {
  func.func @hidden_width(%ctx: !hip.context, %q: memref<8x?x6144xf16, 1>,
      %k: memref<8x?x1024xf16, 1>, %hidden: memref<8x?x5121xf16, 1>,
      %kv: memref<8x4x65600x256xf16, 1>, %len: memref<8xi32, 1>,
      %cap: memref<i32, 1>, %w: memref<640x5120xbf16, 1>,
      %n: memref<128xbf16, 1>, %rope: memref<262144x32xf32, 1>,
      %pos: memref<3x8x?xi64, 1>) {
    // expected-error@+1 {{requires matching B/S}}
    "hip.qwen_qsa_gqa"(%ctx, %q, %k, %k, %kv, %kv, %len, %cap, %hidden, %w, %n, %n, %n,
        %rope, %rope, %pos, %q, %kv, %kv) :
      (!hip.context, memref<8x?x6144xf16, 1>, memref<8x?x1024xf16, 1>, memref<8x?x1024xf16, 1>,
       memref<8x4x65600x256xf16, 1>, memref<8x4x65600x256xf16, 1>, memref<8xi32, 1>, memref<i32, 1>,
       memref<8x?x5121xf16, 1>, memref<640x5120xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>,
       memref<262144x32xf32, 1>, memref<262144x32xf32, 1>, memref<3x8x?xi64, 1>, memref<8x?x6144xf16, 1>,
       memref<8x4x65600x256xf16, 1>, memref<8x4x65600x256xf16, 1>) -> ()
    return
  }
}

// -----
module {
  func.func @selector_dtype(%ctx: !hip.context, %q: memref<8x?x6144xf16, 1>,
      %k: memref<8x?x1024xf16, 1>, %hidden: memref<8x?x5120xf16, 1>,
      %kv: memref<8x4x65600x256xf16, 1>, %len: memref<8xi32, 1>,
      %cap: memref<i32, 1>, %w: memref<640x5120xf16, 1>,
      %n: memref<128xbf16, 1>, %rope: memref<262144x32xf32, 1>,
      %pos: memref<3x8x?xi64, 1>) {
    // expected-error@+1 {{requires frozen BF16 Qwen projection/three norms}}
    "hip.qwen_qsa_gqa"(%ctx, %q, %k, %k, %kv, %kv, %len, %cap, %hidden, %w, %n, %n, %n,
        %rope, %rope, %pos, %q, %kv, %kv) :
      (!hip.context, memref<8x?x6144xf16, 1>, memref<8x?x1024xf16, 1>, memref<8x?x1024xf16, 1>,
       memref<8x4x65600x256xf16, 1>, memref<8x4x65600x256xf16, 1>, memref<8xi32, 1>, memref<i32, 1>,
       memref<8x?x5120xf16, 1>, memref<640x5120xf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>,
       memref<262144x32xf32, 1>, memref<262144x32xf32, 1>, memref<3x8x?xi64, 1>, memref<8x?x6144xf16, 1>,
       memref<8x4x65600x256xf16, 1>, memref<8x4x65600x256xf16, 1>) -> ()
    return
  }
}

// -----
module {
  func.func @position_axes(%ctx: !hip.context, %q: memref<8x?x6144xf16, 1>,
      %k: memref<8x?x1024xf16, 1>, %hidden: memref<8x?x5120xf16, 1>,
      %kv: memref<8x4x65600x256xf16, 1>, %len: memref<8xi32, 1>,
      %cap: memref<i32, 1>, %w: memref<640x5120xbf16, 1>,
      %n: memref<128xbf16, 1>, %rope: memref<262144x32xf32, 1>,
      %pos: memref<1x8x?xi64, 1>) {
    // expected-error@+1 {{requires int32 lengths}}
    "hip.qwen_qsa_gqa"(%ctx, %q, %k, %k, %kv, %kv, %len, %cap, %hidden, %w, %n, %n, %n,
        %rope, %rope, %pos, %q, %kv, %kv) :
      (!hip.context, memref<8x?x6144xf16, 1>, memref<8x?x1024xf16, 1>, memref<8x?x1024xf16, 1>,
       memref<8x4x65600x256xf16, 1>, memref<8x4x65600x256xf16, 1>, memref<8xi32, 1>, memref<i32, 1>,
       memref<8x?x5120xf16, 1>, memref<640x5120xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>,
       memref<262144x32xf32, 1>, memref<262144x32xf32, 1>, memref<1x8x?xi64, 1>, memref<8x?x6144xf16, 1>,
       memref<8x4x65600x256xf16, 1>, memref<8x4x65600x256xf16, 1>) -> ()
    return
  }
}

// -----
module {
  func.func @capacity(%ctx: !hip.context, %q: memref<8x?x6144xf16, 1>,
      %k: memref<8x?x1024xf16, 1>, %hidden: memref<8x?x5120xf16, 1>,
      %kv: memref<8x4x262145x256xf16, 1>, %len: memref<8xi32, 1>,
      %cap: memref<i32, 1>, %w: memref<640x5120xbf16, 1>,
      %n: memref<128xbf16, 1>, %rope: memref<262144x32xf32, 1>,
      %pos: memref<3x8x?xi64, 1>) {
    // expected-error@+1 {{requires capacity through 262144}}
    "hip.qwen_qsa_gqa"(%ctx, %q, %k, %k, %kv, %kv, %len, %cap, %hidden, %w, %n, %n, %n,
        %rope, %rope, %pos, %q, %kv, %kv) :
      (!hip.context, memref<8x?x6144xf16, 1>, memref<8x?x1024xf16, 1>, memref<8x?x1024xf16, 1>,
       memref<8x4x262145x256xf16, 1>, memref<8x4x262145x256xf16, 1>, memref<8xi32, 1>, memref<i32, 1>,
       memref<8x?x5120xf16, 1>, memref<640x5120xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>,
       memref<262144x32xf32, 1>, memref<262144x32xf32, 1>, memref<3x8x?xi64, 1>, memref<8x?x6144xf16, 1>,
       memref<8x4x262145x256xf16, 1>, memref<8x4x262145x256xf16, 1>) -> ()
    return
  }
}

// -----
module {
  func.func @batch(%ctx: !hip.context, %q: memref<9x?x6144xf16, 1>,
      %k: memref<9x?x1024xf16, 1>, %hidden: memref<9x?x5120xf16, 1>,
      %kv: memref<9x4x65600x256xf16, 1>, %len: memref<9xi32, 1>,
      %cap: memref<i32, 1>, %w: memref<640x5120xbf16, 1>,
      %n: memref<128xbf16, 1>, %rope: memref<262144x32xf32, 1>,
      %pos: memref<3x9x?xi64, 1>) {
    // expected-error@+1 {{requires positive sequence length and batch 1 through 8}}
    "hip.qwen_qsa_gqa"(%ctx, %q, %k, %k, %kv, %kv, %len, %cap, %hidden, %w, %n, %n, %n,
        %rope, %rope, %pos, %q, %kv, %kv) :
      (!hip.context, memref<9x?x6144xf16, 1>, memref<9x?x1024xf16, 1>, memref<9x?x1024xf16, 1>,
       memref<9x4x65600x256xf16, 1>, memref<9x4x65600x256xf16, 1>, memref<9xi32, 1>, memref<i32, 1>,
       memref<9x?x5120xf16, 1>, memref<640x5120xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>, memref<128xbf16, 1>,
       memref<262144x32xf32, 1>, memref<262144x32xf32, 1>, memref<3x9x?xi64, 1>, memref<9x?x6144xf16, 1>,
       memref<9x4x65600x256xf16, 1>, memref<9x4x65600x256xf16, 1>) -> ()
    return
  }
}
