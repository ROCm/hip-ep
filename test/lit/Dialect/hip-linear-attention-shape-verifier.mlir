// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics %s

func.func @wrong_state_shape(
    %ctx: !hip.context,
    %query: tensor<1x4x4096xf16>,
    %key: tensor<1x4x1024xf16>,
    %value: tensor<1x4x1024xf16>,
    %output: tensor<1x4x4096xf16>,
    %state: tensor<1x8x64x128xf16>) {
  // expected-error @+1 {{'hip.linear_attention' op present_state dimension 2 must be 128}}
  %result:2 = hip.linear_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x4x4096xf16>, tensor<1x4x1024xf16>,
          tensor<1x4x1024xf16>)
      outs(%output, %state :
           tensor<1x4x4096xf16>, tensor<1x8x64x128xf16>)
      {q_num_heads = 32 : i64, kv_num_heads = 8 : i64}
      : tensor<1x4x4096xf16>, tensor<1x8x64x128xf16>
  return
}

func.func @incompatible_head_counts(
    %ctx: !hip.context,
    %query: tensor<1x4x3840xf16>,
    %key: tensor<1x4x1024xf16>,
    %value: tensor<1x4x1024xf16>,
    %output: tensor<1x4x3840xf16>,
    %state: tensor<1x8x128x128xf16>) {
  // expected-error @+1 {{'hip.linear_attention' op linear_attention q_num_heads and kv_num_heads must divide one another, got 30 and 8}}
  %result:2 = hip.linear_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x4x3840xf16>, tensor<1x4x1024xf16>,
          tensor<1x4x1024xf16>)
      outs(%output, %state :
           tensor<1x4x3840xf16>, tensor<1x8x128x128xf16>)
      {q_num_heads = 30 : i64, kv_num_heads = 8 : i64}
      : tensor<1x4x3840xf16>, tensor<1x8x128x128xf16>
  return
}

// -----

func.func @output_hidden_product_overflow(
    %ctx: !hip.context,
    %query: tensor<1x1x2xf16>,
    %key: tensor<1x1x1xf16>,
    %value: tensor<1x1x9223372036854775807xf16>,
    %output: tensor<1x1x?xf16>,
    %state: tensor<1x1x1x?xf16>) {
  // expected-error @+1 {{'hip.linear_attention' op linear_attention inferred output hidden extent is out of range}}
  %result:2 = hip.linear_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x1x2xf16>, tensor<1x1x1xf16>,
          tensor<1x1x9223372036854775807xf16>)
      outs(%output, %state :
           tensor<1x1x?xf16>, tensor<1x1x1x?xf16>)
      {q_num_heads = 2 : i64, kv_num_heads = 1 : i64}
      : tensor<1x1x?xf16>, tensor<1x1x1x?xf16>
  return
}

// -----

func.func @output_hidden_product_i64_boundary(
    %ctx: !hip.context,
    %query: tensor<1x1x2xf16>,
    %key: tensor<1x1x1xf16>,
    %value: tensor<1x1x4611686018427387903xf16>,
    %output: tensor<1x1x9223372036854775806xf16>,
    %state: tensor<1x1x1x4611686018427387903xf16>) {
  %result:2 = hip.linear_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x1x2xf16>, tensor<1x1x1xf16>,
          tensor<1x1x4611686018427387903xf16>)
      outs(%output, %state :
           tensor<1x1x9223372036854775806xf16>,
           tensor<1x1x1x4611686018427387903xf16>)
      {q_num_heads = 2 : i64, kv_num_heads = 1 : i64}
      : tensor<1x1x9223372036854775806xf16>,
        tensor<1x1x1x4611686018427387903xf16>
  return
}
