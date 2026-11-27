// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The runtime reads hip.matmul_nbits_group's zero points and writes a
// producer's quant_outputs by computed offsets with no bounds check, so the
// verifier rejects operands too small for the rows and groups they serve.
//
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @zero_points_at_row_offsets
// CHECK:         hip.matmul_nbits_group
func.func @zero_points_at_row_offsets(%ctx: !hip.context, %A: memref<1x1x64xf16, 1>,
                                      %B: memref<32x2x16xui8, 1>,
                                      %S: memref<32x2xf16, 1>, %Z: memref<32xui8, 1>,
                                      %a: memref<1x1x24xf16, 1>,
                                      %b: memref<1x1x8xf16, 1>) {
  hip.matmul_nbits_group(%ctx) ins(%A : memref<1x1x64xf16, 1>)
      weights(%B, %B : memref<32x2x16xui8, 1>, memref<32x2x16xui8, 1>)
      scales(%S, %S : memref<32x2xf16, 1>, memref<32x2xf16, 1>)
      zero_points(%Z, %Z : memref<32xui8, 1>, memref<32xui8, 1>)
      outs(%a, %b : memref<1x1x24xf16, 1>, memref<1x1x8xf16, 1>)
      {K = 64 : i64, N = [24, 8], bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, row_offsets = array<i64: 0, 24>}
  return
}

// -----

// Two 4-bit zero points per byte: 16 bytes hold 16 rows of 2 blocks.
func.func @zero_points_too_few_rows(%ctx: !hip.context, %A: memref<1x1x64xf16, 1>,
                                    %B: memref<32x2x16xui8, 1>,
                                    %S: memref<32x2xf16, 1>, %Z: memref<16x1xui8, 1>,
                                    %out: memref<1x1x32xf16, 1>) {
  // expected-error @+1 {{member 0 reads zero-point rows [0, 32) of an operand with 16 rows of 1 bytes}}
  hip.matmul_nbits_group(%ctx) ins(%A : memref<1x1x64xf16, 1>)
      weights(%B : memref<32x2x16xui8, 1>)
      scales(%S : memref<32x2xf16, 1>)
      zero_points(%Z : memref<16x1xui8, 1>)
      outs(%out : memref<1x1x32xf16, 1>)
      {K = 64 : i64, N = [32], bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64}
  return
}

// -----

func.func @zero_points_past_row_offset(%ctx: !hip.context, %A: memref<1x1x64xf16, 1>,
                                       %B: memref<64x2x16xui8, 1>,
                                       %S: memref<64x2xf16, 1>, %Z: memref<32xui8, 1>,
                                       %a: memref<1x1x16xf16, 1>,
                                       %b: memref<1x1x16xf16, 1>) {
  // expected-error @+1 {{member 1 reads zero-point rows [24, 40) of an operand with 32 rows of 1 bytes}}
  hip.matmul_nbits_group(%ctx) ins(%A : memref<1x1x64xf16, 1>)
      weights(%B, %B : memref<64x2x16xui8, 1>, memref<64x2x16xui8, 1>)
      scales(%S, %S : memref<64x2xf16, 1>, memref<64x2xf16, 1>)
      zero_points(%Z, %Z : memref<32xui8, 1>, memref<32xui8, 1>)
      outs(%a, %b : memref<1x1x16xf16, 1>, memref<1x1x16xf16, 1>)
      {K = 64 : i64, N = [16, 16], bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, row_offsets = array<i64: 0, 24>}
  return
}

// -----

// CHECK-LABEL: func.func @norm_quant_outputs
// CHECK:         hip.skip_rms_norm
func.func @norm_quant_outputs(%ctx: !hip.context, %x: memref<1x1x64xf16, 1>,
                              %r: memref<1x1x64xf16, 1>, %g: memref<64xf16, 1>,
                              %y: memref<1x1x64xf16, 1>,
                              %q: memref<1x1x64xi8, 1>, %s: memref<1x1x2xf32, 1>) {
  hip.skip_rms_norm(%ctx)
      ins(%x, %r, %g : memref<1x1x64xf16, 1>, memref<1x1x64xf16, 1>, memref<64xf16, 1>)
      outs(%y : memref<1x1x64xf16, 1>)
      quant_outs(%q, %s : memref<1x1x64xi8, 1>, memref<1x1x2xf32, 1>)
      {epsilon = 1.0e-05 : f32, quant_block_size = 32 : i64}
  return
}

// -----

func.func @norm_quant_wrong_type(%ctx: !hip.context, %x: memref<1x1x64xf16, 1>,
                                 %r: memref<1x1x64xf16, 1>, %g: memref<64xf16, 1>,
                                 %y: memref<1x1x64xf16, 1>,
                                 %q: memref<1x1x64xf16, 1>, %s: memref<1x1x2xf32, 1>) {
  // expected-error @+1 {{a_quant must be i8 shaped like the output it quantizes}}
  hip.skip_rms_norm(%ctx)
      ins(%x, %r, %g : memref<1x1x64xf16, 1>, memref<1x1x64xf16, 1>, memref<64xf16, 1>)
      outs(%y : memref<1x1x64xf16, 1>)
      quant_outs(%q, %s : memref<1x1x64xf16, 1>, memref<1x1x2xf32, 1>)
      {epsilon = 1.0e-05 : f32, quant_block_size = 32 : i64}
  return
}

// -----

func.func @norm_quant_short_row(%ctx: !hip.context, %x: memref<1x1x64xf16, 1>,
                                %r: memref<1x1x64xf16, 1>, %g: memref<64xf16, 1>,
                                %y: memref<1x1x64xf16, 1>,
                                %q: memref<1x1x32xi8, 1>, %s: memref<1x1x2xf32, 1>) {
  // expected-error @+1 {{a_quant must be i8 shaped like the output it quantizes}}
  hip.skip_rms_norm(%ctx)
      ins(%x, %r, %g : memref<1x1x64xf16, 1>, memref<1x1x64xf16, 1>, memref<64xf16, 1>)
      outs(%y : memref<1x1x64xf16, 1>)
      quant_outs(%q, %s : memref<1x1x32xi8, 1>, memref<1x1x2xf32, 1>)
      {epsilon = 1.0e-05 : f32, quant_block_size = 32 : i64}
  return
}

// -----

func.func @norm_quant_wrong_scales(%ctx: !hip.context, %x: memref<1x1x64xf16, 1>,
                                   %r: memref<1x1x64xf16, 1>, %g: memref<64xf16, 1>,
                                   %y: memref<1x1x64xf16, 1>,
                                   %q: memref<1x1x64xi8, 1>, %s: memref<1x1x1xf32, 1>) {
  // expected-error @+1 {{a_scale must be f32 with one entry per quant_block_size run of the quantized output}}
  hip.skip_rms_norm(%ctx)
      ins(%x, %r, %g : memref<1x1x64xf16, 1>, memref<1x1x64xf16, 1>, memref<64xf16, 1>)
      outs(%y : memref<1x1x64xf16, 1>)
      quant_outs(%q, %s : memref<1x1x64xi8, 1>, memref<1x1x1xf32, 1>)
      {epsilon = 1.0e-05 : f32, quant_block_size = 32 : i64}
  return
}

// -----

func.func @norm_quant_partial_block(%ctx: !hip.context, %x: memref<1x1x48xf16, 1>,
                                    %r: memref<1x1x48xf16, 1>, %g: memref<48xf16, 1>,
                                    %y: memref<1x1x48xf16, 1>,
                                    %q: memref<1x1x48xi8, 1>, %s: memref<1x1x1xf32, 1>) {
  // expected-error @+1 {{quantized width 48 is not a multiple of quant_block_size 32}}
  hip.skip_rms_norm(%ctx)
      ins(%x, %r, %g : memref<1x1x48xf16, 1>, memref<1x1x48xf16, 1>, memref<48xf16, 1>)
      outs(%y : memref<1x1x48xf16, 1>)
      quant_outs(%q, %s : memref<1x1x48xi8, 1>, memref<1x1x1xf32, 1>)
      {epsilon = 1.0e-05 : f32, quant_block_size = 32 : i64}
  return
}

// -----

func.func @norm_quant_block_64(%ctx: !hip.context, %x: memref<1x1x128xf16, 1>,
                               %r: memref<1x1x128xf16, 1>, %g: memref<128xf16, 1>,
                               %y: memref<1x1x128xf16, 1>,
                               %q: memref<1x1x128xi8, 1>, %s: memref<1x1x2xf32, 1>) {
  // expected-error @+1 {{quant_outputs require quant_block_size 32, got 64}}
  hip.skip_rms_norm(%ctx)
      ins(%x, %r, %g : memref<1x1x128xf16, 1>, memref<1x1x128xf16, 1>, memref<128xf16, 1>)
      outs(%y : memref<1x1x128xf16, 1>)
      quant_outs(%q, %s : memref<1x1x128xi8, 1>, memref<1x1x2xf32, 1>)
      {epsilon = 1.0e-05 : f32, quant_block_size = 64 : i64}
  return
}

// -----

func.func @norm_quant_no_block(%ctx: !hip.context, %x: memref<1x1x64xf16, 1>,
                               %r: memref<1x1x64xf16, 1>, %g: memref<64xf16, 1>,
                               %y: memref<1x1x64xf16, 1>,
                               %q: memref<1x1x64xi8, 1>, %s: memref<1x1x2xf32, 1>) {
  // expected-error @+1 {{quant_outputs require quant_block_size 32, got 0}}
  hip.skip_rms_norm(%ctx)
      ins(%x, %r, %g : memref<1x1x64xf16, 1>, memref<1x1x64xf16, 1>, memref<64xf16, 1>)
      outs(%y : memref<1x1x64xf16, 1>)
      quant_outs(%q, %s : memref<1x1x64xi8, 1>, memref<1x1x2xf32, 1>)
      {epsilon = 1.0e-05 : f32}
  return
}

// -----

func.func @pre_mix_quant_wrong_scales(%ctx: !hip.context, %x: memref<1x1x64xf16, 1>,
                                      %g: memref<1x1x64xf16, 1>,
                                      %y: memref<1x1x32xf16, 1>,
                                      %q: memref<1x1x32xi8, 1>, %s: memref<1x1x2xf32, 1>) {
  // expected-error @+1 {{a_scale must be f32 with one entry per quant_block_size run of the quantized output}}
  hip.hyper_connection_pre_mix(%ctx)
      ins(%x, %g : memref<1x1x64xf16, 1>, memref<1x1x64xf16, 1>)
      outs(%y : memref<1x1x32xf16, 1>)
      quant_outs(%q, %s : memref<1x1x32xi8, 1>, memref<1x1x2xf32, 1>)
      {branches = 2 : i64, flattened = 1 : i64, gate_layout = "feature",
       hidden = 32 : i64, reduction_scale = 5.000000e-01 : f32,
       quant_block_size = 32 : i64}
  return
}
