// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics %s

func.func @hidden_mismatch(
    %ctx: !hip.context,
    %query: tensor<1x8x128xf16>,
    %key: tensor<1x16x128xf16>,
    %value: tensor<1x16x64xf16>,
    %output: tensor<1x8x128xf16>) {
  // expected-error @+1 {{'hip.multi_head_attention' op multi_head_attention Q/K/V hidden extents must agree}}
  %result = hip.multi_head_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x8x128xf16>, tensor<1x16x128xf16>,
          tensor<1x16x64xf16>)
      outs(%output : tensor<1x8x128xf16>)
      {num_heads = 8 : i64}
      : tensor<1x8x128xf16>
  return
}

func.func @kv_sequence_mismatch(
    %ctx: !hip.context,
    %query: tensor<1x8x128xf16>,
    %key: tensor<1x16x128xf16>,
    %value: tensor<1x15x128xf16>,
    %output: tensor<1x8x128xf16>) {
  // expected-error @+1 {{'hip.multi_head_attention' op multi_head_attention K/V sequence extents must agree}}
  %result = hip.multi_head_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x8x128xf16>, tensor<1x16x128xf16>,
          tensor<1x15x128xf16>)
      outs(%output : tensor<1x8x128xf16>)
      {num_heads = 8 : i64}
      : tensor<1x8x128xf16>
  return
}

func.func @unsupported_present_output(
    %ctx: !hip.context,
    %query: tensor<1x8x128xf16>,
    %key: tensor<1x16x128xf16>,
    %value: tensor<1x16x128xf16>,
    %output: tensor<1x8x128xf16>,
    %presentKey: tensor<1x8x16x16xf16>) {
  // expected-error @+1 {{'hip.multi_head_attention' op default runtime does not support present_key, present_value, or qk outputs}}
  %result:2 = hip.multi_head_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x8x128xf16>, tensor<1x16x128xf16>,
          tensor<1x16x128xf16>)
      outs(%output, %presentKey :
          tensor<1x8x128xf16>, tensor<1x8x16x16xf16>)
      {num_heads = 8 : i64}
      : tensor<1x8x128xf16>, tensor<1x8x16x16xf16>
  return
}

func.func @output_mismatch(
    %ctx: !hip.context,
    %query: tensor<1x8x128xf16>,
    %key: tensor<1x16x128xf16>,
    %value: tensor<1x16x128xf16>,
    %output: tensor<1x7x128xf16>) {
  // expected-error @+1 {{'hip.multi_head_attention' op dim 1 of result mismatch: expected 8 [1, 8, 128] but outs has 7 [1, 7, 128]}}
  %result = hip.multi_head_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x8x128xf16>, tensor<1x16x128xf16>,
          tensor<1x16x128xf16>)
      outs(%output : tensor<1x7x128xf16>)
      {num_heads = 8 : i64}
      : tensor<1x7x128xf16>
  return
}

func.func @unsupported_mask_filter_value(
    %ctx: !hip.context,
    %query: tensor<1x8x128xf16>,
    %key: tensor<1x16x128xf16>,
    %value: tensor<1x16x128xf16>,
    %output: tensor<1x8x128xf16>) {
  // expected-error @+1 {{'hip.multi_head_attention' op default runtime supports only mask_filter_value = -10000}}
  %result = hip.multi_head_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x8x128xf16>, tensor<1x16x128xf16>,
          tensor<1x16x128xf16>)
      outs(%output : tensor<1x8x128xf16>)
      {num_heads = 8 : i64, mask_filter_value = -5.0 : f32}
      : tensor<1x8x128xf16>
  return
}
