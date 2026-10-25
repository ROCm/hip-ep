// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @wrong_output(%ctx: !hip.context, %a: tensor<2x16xf16>,
                        %b: tensor<8x1x8xui8>, %s: tensor<8x1xf16>,
                        %out: tensor<2x16xf16>) -> tensor<2x16xf16> {
  // expected-error @+1 {{dim 1 of result mismatch: expected 8}}
  %r = hip.matmul_nbits(%ctx)
    ins(%a, %b, %s : tensor<2x16xf16>, tensor<8x1x8xui8>, tensor<8x1xf16>)
    outs(%out : tensor<2x16xf16>)
    {K = 16 : i64, N = 8 : i64, bits = 4 : i64, block_size = 16 : i64,
     accuracy_level = 0 : i64, zp_elem_size = 2 : i64} : tensor<2x16xf16>
  return %r : tensor<2x16xf16>
}

// -----

func.func @wrong_k(%ctx: !hip.context, %a: tensor<2x32xf16>,
                   %b: tensor<8x1x8xui8>, %s: tensor<8x1xf16>,
                   %out: tensor<2x8xf16>) -> tensor<2x8xf16> {
  // expected-error @+1 {{A's trailing extent must equal K}}
  %r = hip.matmul_nbits(%ctx)
    ins(%a, %b, %s : tensor<2x32xf16>, tensor<8x1x8xui8>, tensor<8x1xf16>)
    outs(%out : tensor<2x8xf16>)
    {K = 16 : i64, N = 8 : i64, bits = 4 : i64, block_size = 16 : i64,
     accuracy_level = 0 : i64, zp_elem_size = 2 : i64} : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// -----

func.func @rank_zero(%ctx: !hip.context, %a: tensor<f16>,
                     %b: tensor<8x1x8xui8>, %s: tensor<8x1xf16>,
                     %out: tensor<8xf16>) -> tensor<8xf16> {
  // expected-error @+1 {{A must have rank at least 1}}
  %r = hip.matmul_nbits(%ctx)
    ins(%a, %b, %s : tensor<f16>, tensor<8x1x8xui8>, tensor<8x1xf16>)
    outs(%out : tensor<8xf16>)
    {K = 16 : i64, N = 8 : i64, bits = 4 : i64, block_size = 16 : i64,
     accuracy_level = 0 : i64, zp_elem_size = 2 : i64} : tensor<8xf16>
  return %r : tensor<8xf16>
}
