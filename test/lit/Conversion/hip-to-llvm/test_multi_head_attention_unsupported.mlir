// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// These forms are valid HIP IR but are not implemented by the default wrapper.
// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm --split-input-file --verify-diagnostics %s

func.func @packed_qkv(%ctx: !hip.context,
    %qkv: memref<1x8x8x3x16xf16, 1>, %out: memref<1x8x128xf16, 1>) {
  // expected-error @+2 {{default runtime requires separate query, key, and value inputs}}
  // expected-error @+1 {{failed to legalize operation 'hip.multi_head_attention'}}
  hip.multi_head_attention(%ctx)
      ins(%qkv : memref<1x8x8x3x16xf16, 1>)
      outs(%out : memref<1x8x128xf16, 1>) {num_heads = 8 : i64}
  return
}

// -----

func.func @present_key_only(%ctx: !hip.context,
    %q: memref<1x8x128xf16, 1>, %k: memref<1x16x128xf16, 1>,
    %v: memref<1x16x128xf16, 1>, %out: memref<1x8x128xf16, 1>,
    %pk: memref<1x8x16x16xf16, 1>) {
  // expected-error @+2 {{default runtime does not support present_key, present_value, or qk outputs}}
  // expected-error @+1 {{failed to legalize operation 'hip.multi_head_attention'}}
  hip.multi_head_attention(%ctx)
      ins(%q, %k, %v : memref<1x8x128xf16, 1>, memref<1x16x128xf16, 1>, memref<1x16x128xf16, 1>)
      outs(%out, %pk : memref<1x8x128xf16, 1>, memref<1x8x16x16xf16, 1>)
      {num_heads = 8 : i64}
  return
}

// -----

func.func @nondefault_mask_filter(%ctx: !hip.context,
    %q: memref<1x8x128xf16, 1>, %k: memref<1x16x128xf16, 1>,
    %v: memref<1x16x128xf16, 1>, %out: memref<1x8x128xf16, 1>) {
  // expected-error @+2 {{default runtime supports only mask_filter_value = -10000}}
  // expected-error @+1 {{failed to legalize operation 'hip.multi_head_attention'}}
  hip.multi_head_attention(%ctx)
      ins(%q, %k, %v : memref<1x8x128xf16, 1>, memref<1x16x128xf16, 1>, memref<1x16x128xf16, 1>)
      outs(%out : memref<1x8x128xf16, 1>)
      {num_heads = 8 : i64, mask_filter_value = -5.0 : f32}
  return
}

// -----

func.func @different_value_width(%ctx: !hip.context,
    %q: memref<1x8x128xf16, 1>, %k: memref<1x16x128xf16, 1>,
    %v: memref<1x16x64xf16, 1>, %out: memref<1x8x64xf16, 1>) {
  // expected-error @+2 {{default runtime requires equal Q/V hidden extents}}
  // expected-error @+1 {{failed to legalize operation 'hip.multi_head_attention'}}
  hip.multi_head_attention(%ctx)
      ins(%q, %k, %v : memref<1x8x128xf16, 1>, memref<1x16x128xf16, 1>, memref<1x16x64xf16, 1>)
      outs(%out : memref<1x8x64xf16, 1>) {num_heads = 8 : i64}
  return
}

// -----

func.func @f32(%ctx: !hip.context,
    %q: memref<1x8x128xf32, 1>, %k: memref<1x16x128xf32, 1>,
    %v: memref<1x16x128xf32, 1>, %out: memref<1x8x128xf32, 1>) {
  // expected-error @+2 {{default runtime requires fp16 Q/K/V and output}}
  // expected-error @+1 {{failed to legalize operation 'hip.multi_head_attention'}}
  hip.multi_head_attention(%ctx)
      ins(%q, %k, %v : memref<1x8x128xf32, 1>, memref<1x16x128xf32, 1>, memref<1x16x128xf32, 1>)
      outs(%out : memref<1x8x128xf32, 1>) {num_heads = 8 : i64}
  return
}

// -----

func.func @bnsh_kv(%ctx: !hip.context,
    %q: memref<1x8x128xf16, 1>, %k: memref<1x8x16x16xf16, 1>,
    %v: memref<1x8x16x16xf16, 1>, %out: memref<1x8x128xf16, 1>) {
  // expected-error @+2 {{default runtime requires rank-3 Q/K/V}}
  // expected-error @+1 {{failed to legalize operation 'hip.multi_head_attention'}}
  hip.multi_head_attention(%ctx)
      ins(%q, %k, %v : memref<1x8x128xf16, 1>, memref<1x8x16x16xf16, 1>, memref<1x8x16x16xf16, 1>)
      outs(%out : memref<1x8x128xf16, 1>) {num_heads = 8 : i64}
  return
}

// -----

func.func @projection_bias(%ctx: !hip.context,
    %q: memref<1x8x128xf16, 1>, %k: memref<1x16x128xf16, 1>,
    %v: memref<1x16x128xf16, 1>, %bias: memref<384xf16, 1>,
    %out: memref<1x8x128xf16, 1>) {
  // expected-error @+2 {{default runtime does not support bias, masks, past/cache inputs, or cache indirection}}
  // expected-error @+1 {{failed to legalize operation 'hip.multi_head_attention'}}
  hip.multi_head_attention(%ctx)
      ins(%q, %k, %v, %bias : memref<1x8x128xf16, 1>, memref<1x16x128xf16, 1>, memref<1x16x128xf16, 1>, memref<384xf16, 1>)
      outs(%out : memref<1x8x128xf16, 1>) {num_heads = 8 : i64}
  return
}
