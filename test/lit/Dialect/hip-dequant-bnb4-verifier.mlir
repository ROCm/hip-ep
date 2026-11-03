// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.dequant_bnb4 reads ui8, reads f32 absmax, and writes f32 [N, K].
// The HIP-to-LLVM lowering forwards the pointers without checking those
// sizes, so a mismatched operand overruns the output buffer.
//
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @dequant_bnb4_tensor
// CHECK:         hip.dequant_bnb4
func.func @dequant_bnb4_tensor(%ctx: !hip.context, %packed: tensor<16xui8>,
                               %absmax: tensor<2xf32>, %init: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
  %r = hip.dequant_bnb4(%ctx) ins(%packed, %absmax :
        tensor<16xui8>, tensor<2xf32>) outs(%init : tensor<4x8xf32>)
        {K = 8 : i64, N = 4 : i64, block_size = 16 : i64, quant_type = 1 : i64}
        : tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @dequant_bnb4_memref
// CHECK:         hip.dequant_bnb4
func.func @dequant_bnb4_memref(%ctx: !hip.context, %packed: memref<16xui8, 1>,
                               %absmax: memref<2xf32, 1>,
                               %out: memref<4x8xf32, 1>) {
  hip.dequant_bnb4(%ctx) ins(%packed, %absmax :
        memref<16xui8, 1>, memref<2xf32, 1>)
      outs(%out : memref<4x8xf32, 1>)
      {K = 8 : i64, N = 4 : i64, block_size = 16 : i64, quant_type = 0 : i64}
  return
}

// -----

func.func @dequant_bnb4_f16_output(%ctx: !hip.context, %packed: tensor<16xui8>,
                                   %absmax: tensor<2xf32>,
                                   %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error @+1 {{absmax and output must be f32}}
  %r = hip.dequant_bnb4(%ctx) ins(%packed, %absmax :
        tensor<16xui8>, tensor<2xf32>) outs(%init : tensor<4x8xf16>)
        {K = 8 : i64, N = 4 : i64, block_size = 16 : i64, quant_type = 1 : i64}
        : tensor<4x8xf16>
  return %r : tensor<4x8xf16>
}

// -----

func.func @dequant_bnb4_f16_output_memref(%ctx: !hip.context,
                                          %packed: memref<16xui8, 1>,
                                          %absmax: memref<2xf32, 1>,
                                          %out: memref<4x8xf16, 1>) {
  // expected-error @below {{absmax and output must be f32}}
  hip.dequant_bnb4(%ctx) ins(%packed, %absmax :
        memref<16xui8, 1>, memref<2xf32, 1>)
      outs(%out : memref<4x8xf16, 1>)
      {K = 8 : i64, N = 4 : i64, block_size = 16 : i64, quant_type = 1 : i64}
  return
}

// -----

func.func @dequant_bnb4_short_packed(%ctx: !hip.context,
                                     %packed: tensor<8xui8>,
                                     %absmax: tensor<2xf32>,
                                     %init: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error @below {{packed length must be N * K / 2}}
  %r = hip.dequant_bnb4(%ctx) ins(%packed, %absmax :
        tensor<8xui8>, tensor<2xf32>) outs(%init : tensor<4x8xf32>)
        {K = 8 : i64, N = 4 : i64, block_size = 16 : i64, quant_type = 1 : i64}
        : tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// -----

func.func @dequant_bnb4_short_packed_memref(%ctx: !hip.context,
                                            %packed: memref<8xui8, 1>,
                                            %absmax: memref<2xf32, 1>,
                                            %out: memref<4x8xf32, 1>) {
  // expected-error @below {{packed length must be N * K / 2}}
  hip.dequant_bnb4(%ctx) ins(%packed, %absmax :
        memref<8xui8, 1>, memref<2xf32, 1>)
      outs(%out : memref<4x8xf32, 1>)
      {K = 8 : i64, N = 4 : i64, block_size = 16 : i64, quant_type = 1 : i64}
  return
}

// -----

func.func @dequant_bnb4_bad_block(%ctx: !hip.context, %packed: tensor<16xui8>,
                                  %absmax: tensor<2xf32>, %init: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
  // expected-error @below {{block_size must be 16, 32, 64, 128, or 256}}
  %r = hip.dequant_bnb4(%ctx) ins(%packed, %absmax :
        tensor<16xui8>, tensor<2xf32>) outs(%init : tensor<4x8xf32>)
        {K = 8 : i64, N = 4 : i64, block_size = 8 : i64, quant_type = 1 : i64}
        : tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}
