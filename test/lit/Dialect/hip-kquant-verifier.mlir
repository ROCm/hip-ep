// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.quantize_q8_1, hip.skip_rms_norm_q8_1, hip.kquant_matmul,
// hip.kquant_matmul_q8_1, hip.kquant_swiglu_q8_1 and hip.token_major_gqa: their
// printed forms parse back, and their verifiers.
//
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @projections
// CHECK: hip.skip_rms_norm_q8_1
// CHECK: hip.kquant_matmul_q8_1
// CHECK-SAME: format = "q4_k"
// CHECK: hip.kquant_swiglu_q8_1
// CHECK-SAME: gate_format = "q4_k"
// CHECK-SAME: up_format = "q6_k"
// CHECK: hip.quantize_q8_1
// CHECK: hip.kquant_matmul
// CHECK-SAME: format = "q5_k"
func.func @projections(
    %ctx: !hip.context, %x: tensor<1x1x4x512xf32>, %skip: tensor<1x1x4x512xf32>,
    %gamma: tensor<512xf32>, %sum: tensor<1x1x4x512xf32>,
    %q: tensor<1x1x4x576xui8>, %q4: tensor<864xui8>, %q5: tensor<1056xui8>,
    %q6: tensor<1260xui8>, %y: tensor<1x1x4x3xf32>)
    -> (tensor<1x1x4x3xf32>, tensor<1x1x4x3xf32>, tensor<1x1x4x3xf32>) {
  %s:2 = hip.skip_rms_norm_q8_1(%ctx)
    ins(%x, %skip, %gamma : tensor<1x1x4x512xf32>, tensor<1x1x4x512xf32>,
        tensor<512xf32>)
    outs(%sum, %q : tensor<1x1x4x512xf32>, tensor<1x1x4x576xui8>)
    {K = 512 : i64, epsilon = 1.0e-05 : f32}
      : tensor<1x1x4x512xf32>, tensor<1x1x4x576xui8>
  %a = hip.kquant_matmul_q8_1(%ctx)
    ins(%s#1, %q4 : tensor<1x1x4x576xui8>, tensor<864xui8>)
    outs(%y : tensor<1x1x4x3xf32>)
    {K = 512 : i64, N = 3 : i64, format = "q4_k"} : tensor<1x1x4x3xf32>
  %b = hip.kquant_swiglu_q8_1(%ctx)
    ins(%s#1, %q4, %q6 : tensor<1x1x4x576xui8>, tensor<864xui8>,
        tensor<1260xui8>)
    outs(%y : tensor<1x1x4x3xf32>)
    {K = 512 : i64, N = 3 : i64, gate_format = "q4_k", up_format = "q6_k"}
      : tensor<1x1x4x3xf32>
  %c = hip.quantize_q8_1(%ctx)
    ins(%s#0 : tensor<1x1x4x512xf32>) outs(%q : tensor<1x1x4x576xui8>)
    {K = 512 : i64} : tensor<1x1x4x576xui8>
  %d = hip.kquant_matmul(%ctx)
    ins(%x, %q5 : tensor<1x1x4x512xf32>, tensor<1056xui8>)
    outs(%y : tensor<1x1x4x3xf32>)
    {K = 512 : i64, N = 3 : i64, format = "q5_k"} : tensor<1x1x4x3xf32>
  return %a, %b, %d : tensor<1x1x4x3xf32>, tensor<1x1x4x3xf32>,
                      tensor<1x1x4x3xf32>
}

// -----

// CHECK-LABEL: func.func @attention
// CHECK: hip.token_major_gqa
// CHECK-SAME: rows(
func.func @attention(
    %ctx: !hip.context, %q: tensor<1x?x8x64xf32>, %k: tensor<1x?x2x64xf16>,
    %v: tensor<1x?x2x64xf16>, %mask: tensor<1x1x?x?xf16>,
    %rows: tensor<1x1x1x?xi64>, %o: tensor<1x?x8x64xf32>)
    -> tensor<1x?x8x64xf32> {
  %r = hip.token_major_gqa(%ctx)
    ins(%q, %k, %v, %mask : tensor<1x?x8x64xf32>, tensor<1x?x2x64xf16>,
        tensor<1x?x2x64xf16>, tensor<1x1x?x?xf16>)
    rows(%rows : tensor<1x1x1x?xi64>)
    outs(%o : tensor<1x?x8x64xf32>) {scale = 0.125 : f32}
      : tensor<1x?x8x64xf32>
  return %r : tensor<1x?x8x64xf32>
}

// -----

func.func @wrong_weight_bytes(%ctx: !hip.context, %x: tensor<1x4x512xf32>,
                              %w: tensor<863xui8>, %y: tensor<1x4x3xf32>) {
  // expected-error @+1 {{packed_weights has 863 bytes but q4_k [3, 512] has 864}}
  %r = hip.kquant_matmul(%ctx) ins(%x, %w : tensor<1x4x512xf32>, tensor<863xui8>)
    outs(%y : tensor<1x4x3xf32>) {K = 512 : i64, N = 3 : i64, format = "q4_k"}
      : tensor<1x4x3xf32>
  return
}

// -----

func.func @unknown_format(%ctx: !hip.context, %x: tensor<1x4x512xf32>,
                          %w: tensor<864xui8>, %y: tensor<1x4x3xf32>) {
  // expected-error @+1 {{requires packed_weights format q4_k, q5_k or q6_k}}
  %r = hip.kquant_matmul(%ctx) ins(%x, %w : tensor<1x4x512xf32>, tensor<864xui8>)
    outs(%y : tensor<1x4x3xf32>) {K = 512 : i64, N = 3 : i64, format = "q8_0"}
      : tensor<1x4x3xf32>
  return
}

// -----

// A projection reads the sum field as its Q8_1 producer wrote it.
func.func @sum_mode_mismatch(%ctx: !hip.context, %x: tensor<1x4x512xf32>,
                             %q: tensor<1x4x576xui8>, %w: tensor<864xui8>,
                             %y: tensor<1x4x3xf32>) {
  %p = hip.quantize_q8_1(%ctx) ins(%x : tensor<1x4x512xf32>)
    outs(%q : tensor<1x4x576xui8>) {K = 512 : i64, sum_mode = 2 : i64}
      : tensor<1x4x576xui8>
  // expected-error @+1 {{requires sum_mode and rows_per_token to match the Q8_1 producer's (2, 1)}}
  %r = hip.kquant_matmul_q8_1(%ctx)
    ins(%p, %w : tensor<1x4x576xui8>, tensor<864xui8>)
    outs(%y : tensor<1x4x3xf32>) {K = 512 : i64, N = 3 : i64, format = "q4_k"}
      : tensor<1x4x3xf32>
  return
}

// -----

func.func @q8_row_bytes(%ctx: !hip.context, %x: tensor<1x4x512xf32>,
                        %q: tensor<1x4x512xui8>) {
  // expected-error @+1 {{requires the Q8_1 innermost dimension to be 576 bytes for K=512}}
  %p = hip.quantize_q8_1(%ctx) ins(%x : tensor<1x4x512xf32>)
    outs(%q : tensor<1x4x512xui8>) {K = 512 : i64} : tensor<1x4x512xui8>
  return
}

// -----

func.func @swiglu_q5(%ctx: !hip.context, %q: tensor<1x1x4x576xui8>,
                     %w: tensor<1056xui8>, %y: tensor<1x1x4x3xf32>) {
  // expected-error @+1 {{requires q4_k or q6_k gate and up formats}}
  %r = hip.kquant_swiglu_q8_1(%ctx)
    ins(%q, %w, %w : tensor<1x1x4x576xui8>, tensor<1056xui8>, tensor<1056xui8>)
    outs(%y : tensor<1x1x4x3xf32>)
    {K = 512 : i64, N = 3 : i64, gate_format = "q5_k", up_format = "q5_k"}
      : tensor<1x1x4x3xf32>
  return
}

// -----

func.func @head_dimension(%ctx: !hip.context, %q: tensor<1x4x8x48xf32>,
                          %k: tensor<1x16x2x48xf16>, %mask: tensor<1x1x4x16xf16>,
                          %o: tensor<1x4x8x48xf32>) {
  // expected-error @+1 {{requires D to be 40, 64, 72, 80, 96, 112, 128 or 256}}
  %r = hip.token_major_gqa(%ctx)
    ins(%q, %k, %k, %mask : tensor<1x4x8x48xf32>, tensor<1x16x2x48xf16>,
        tensor<1x16x2x48xf16>, tensor<1x1x4x16xf16>)
    outs(%o : tensor<1x4x8x48xf32>) {scale = 0.125 : f32}
      : tensor<1x4x8x48xf32>
  return
}

// -----

func.func @mask_shape(%ctx: !hip.context, %q: tensor<1x4x8x64xf32>,
                      %k: tensor<1x16x2x64xf16>, %mask: tensor<1x1x4x8xf16>,
                      %o: tensor<1x4x8x64xf32>) {
  // expected-error @+1 {{requires a [1|B, 1, S, C] mask}}
  %r = hip.token_major_gqa(%ctx)
    ins(%q, %k, %k, %mask : tensor<1x4x8x64xf32>, tensor<1x16x2x64xf16>,
        tensor<1x16x2x64xf16>, tensor<1x1x4x8xf16>)
    outs(%o : tensor<1x4x8x64xf32>) {scale = 0.125 : f32}
      : tensor<1x4x8x64xf32>
  return
}
