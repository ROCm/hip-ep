// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// bf16 must not fall through to hip.multi_head_attention. The runtime kernel
// is told only the element size, and a 2-byte buffer is dispatched as f16.
// RUN: hip-mlir-opt --hip-decompose-multi-head-attention --verify-diagnostics %s

func.func @bf16(%ctx: !hip.context, %q: tensor<1x4x16xbf16>,
                %k: tensor<1x4x16xbf16>, %v: tensor<1x4x16xbf16>)
    -> tensor<1x4x16xbf16> {
  %empty = tensor.empty() : tensor<1x4x16xbf16>
  // expected-error @+1 {{bf16 is not supported}}
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x4x16xbf16>,
      tensor<1x4x16xbf16>, tensor<1x4x16xbf16>)
      outs(%empty : tensor<1x4x16xbf16>)
      {num_heads = 2 : i64, scale = 5.000000e-01 : f32} : tensor<1x4x16xbf16>
  return %r : tensor<1x4x16xbf16>
}
