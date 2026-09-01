// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @valid_separate(
    %ctx: !hip.context,
    %query: tensor<1x2x32xf16>,
    %key: tensor<1x3x16xf16>,
    %value: tensor<1x3x16xf16>,
    %past_key: tensor<1x2x4x8xf16>,
    %past_value: tensor<1x2x4x8xf16>,
    %seqlens: tensor<1xi32>,
    %total: tensor<i32>,
    %out: tensor<1x2x32xf16>,
    %present_key: tensor<1x2x5x8xf16>,
    %present_value: tensor<1x2x5x8xf16>)
    -> (tensor<1x2x32xf16>, tensor<1x2x5x8xf16>,
        tensor<1x2x5x8xf16>) {
  %result:3 = hip.gqa(%ctx)
    ins(%query, %key, %value, %past_key, %past_value, %seqlens, %total :
        tensor<1x2x32xf16>, tensor<1x3x16xf16>, tensor<1x3x16xf16>,
        tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>, tensor<1xi32>,
        tensor<i32>)
    outs(%out, %present_key, %present_value :
        tensor<1x2x32xf16>, tensor<1x2x5x8xf16>, tensor<1x2x5x8xf16>)
    {num_heads = 4 : i64, kv_num_heads = 2 : i64}
    : tensor<1x2x32xf16>, tensor<1x2x5x8xf16>,
      tensor<1x2x5x8xf16>
  return %result#0, %result#1, %result#2 :
      tensor<1x2x32xf16>, tensor<1x2x5x8xf16>, tensor<1x2x5x8xf16>
}

// -----

func.func @valid_memref(
    %ctx: !hip.context,
    %query: memref<1x2x32xf16, 1>,
    %key: memref<1x3x16xf16, 1>,
    %value: memref<1x3x16xf16, 1>,
    %past_key: memref<1x2x4x8xf16, 1>,
    %past_value: memref<1x2x4x8xf16, 1>,
    %seqlens: memref<1xi32, 1>,
    %total: memref<i32, 1>,
    %out: memref<1x2x32xf16, 1>,
    %present_key: memref<1x2x5x8xf16, 1>,
    %present_value: memref<1x2x5x8xf16, 1>) {
  hip.gqa(%ctx)
    ins(%query, %key, %value, %past_key, %past_value, %seqlens, %total :
        memref<1x2x32xf16, 1>, memref<1x3x16xf16, 1>,
        memref<1x3x16xf16, 1>, memref<1x2x4x8xf16, 1>,
        memref<1x2x4x8xf16, 1>, memref<1xi32, 1>, memref<i32, 1>)
    outs(%out, %present_key, %present_value :
        memref<1x2x32xf16, 1>, memref<1x2x5x8xf16, 1>,
        memref<1x2x5x8xf16, 1>)
    {num_heads = 4 : i64, kv_num_heads = 2 : i64}
  return
}

// -----

func.func @invalid_head_divisibility(
    %ctx: !hip.context,
    %query: memref<1x2x30xf16, 1>,
    %key: memref<1x3x16xf16, 1>,
    %value: memref<1x3x16xf16, 1>,
    %past_key: memref<1x2x4x8xf16, 1>,
    %past_value: memref<1x2x4x8xf16, 1>,
    %seqlens: memref<1xi32, 1>,
    %total: memref<i32, 1>,
    %out: memref<1x2x30xf16, 1>,
    %present_key: memref<1x2x5x8xf16, 1>,
    %present_value: memref<1x2x5x8xf16, 1>) {
  // expected-error @+1 {{query hidden extent must be divisible by num_heads}}
  hip.gqa(%ctx)
    ins(%query, %key, %value, %past_key, %past_value, %seqlens, %total :
        memref<1x2x30xf16, 1>, memref<1x3x16xf16, 1>,
        memref<1x3x16xf16, 1>, memref<1x2x4x8xf16, 1>,
        memref<1x2x4x8xf16, 1>, memref<1xi32, 1>, memref<i32, 1>)
    outs(%out, %present_key, %present_value :
        memref<1x2x30xf16, 1>, memref<1x2x5x8xf16, 1>,
        memref<1x2x5x8xf16, 1>)
    {num_heads = 4 : i64, kv_num_heads = 2 : i64}
  return
}

// -----

func.func @invalid_present_heads(
    %ctx: !hip.context,
    %query: memref<1x2x32xf16, 1>,
    %key: memref<1x3x16xf16, 1>,
    %value: memref<1x3x16xf16, 1>,
    %past_key: memref<1x3x4x8xf16, 1>,
    %past_value: memref<1x3x4x8xf16, 1>,
    %seqlens: memref<1xi32, 1>,
    %total: memref<i32, 1>,
    %out: memref<1x2x32xf16, 1>,
    %present_key: memref<1x3x5x8xf16, 1>,
    %present_value: memref<1x3x5x8xf16, 1>) {
  // expected-error @+1 {{present_key head count must equal kv_num_heads}}
  hip.gqa(%ctx)
    ins(%query, %key, %value, %past_key, %past_value, %seqlens, %total :
        memref<1x2x32xf16, 1>, memref<1x3x16xf16, 1>,
        memref<1x3x16xf16, 1>, memref<1x3x4x8xf16, 1>,
        memref<1x3x4x8xf16, 1>, memref<1xi32, 1>, memref<i32, 1>)
    outs(%out, %present_key, %present_value :
        memref<1x2x32xf16, 1>, memref<1x3x5x8xf16, 1>,
        memref<1x3x5x8xf16, 1>)
    {num_heads = 4 : i64, kv_num_heads = 2 : i64}
  return
}

// -----

func.func @invalid_result_cardinality(
    %ctx: !hip.context,
    %query: tensor<1x2x32xf16>,
    %key: tensor<1x3x16xf16>,
    %value: tensor<1x3x16xf16>,
    %past_key: tensor<1x2x4x8xf16>,
    %past_value: tensor<1x2x4x8xf16>,
    %seqlens: tensor<1xi32>,
    %total: tensor<i32>,
    %out: tensor<1x2x32xf16>,
    %present_key: tensor<1x2x5x8xf16>,
    %present_value: tensor<1x2x5x8xf16>)
    -> (tensor<1x2x32xf16>, tensor<1x2x5x8xf16>) {
  // expected-error @+1 {{tensor mode requires 3 result(s), got 2}}
  %result:2 = hip.gqa(%ctx)
    ins(%query, %key, %value, %past_key, %past_value, %seqlens, %total :
        tensor<1x2x32xf16>, tensor<1x3x16xf16>, tensor<1x3x16xf16>,
        tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>, tensor<1xi32>,
        tensor<i32>)
    outs(%out, %present_key, %present_value :
        tensor<1x2x32xf16>, tensor<1x2x5x8xf16>, tensor<1x2x5x8xf16>)
    {num_heads = 4 : i64, kv_num_heads = 2 : i64}
    : tensor<1x2x32xf16>, tensor<1x2x5x8xf16>
  return %result#0, %result#1 :
      tensor<1x2x32xf16>, tensor<1x2x5x8xf16>
}

// -----

func.func @unpaired_key_value(
    %ctx: !hip.context,
    %query: tensor<1x2x32xf16>,
    %key: tensor<1x3x16xf16>,
    %seqlens: tensor<1xi32>,
    %total: tensor<i32>,
    %out: tensor<1x2x32xf16>,
    %present_key: tensor<1x2x5x8xf16>,
    %present_value: tensor<1x2x5x8xf16>)
    -> (tensor<1x2x32xf16>, tensor<1x2x5x8xf16>,
        tensor<1x2x5x8xf16>) {
  // expected-error @+1 {{key and value must both be provided or both be omitted}}
  %result:3 = "hip.gqa"(
      %ctx, %query, %key, %seqlens, %total, %out, %present_key, %present_value)
      <{num_heads = 4 : i64, kv_num_heads = 2 : i64,
        operandSegmentSizes =
          array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>}>
      : (!hip.context, tensor<1x2x32xf16>, tensor<1x3x16xf16>,
         tensor<1xi32>, tensor<i32>, tensor<1x2x32xf16>,
         tensor<1x2x5x8xf16>, tensor<1x2x5x8xf16>)
      -> (tensor<1x2x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>)
  return %result#0, %result#1, %result#2 :
      tensor<1x2x32xf16>, tensor<1x2x5x8xf16>, tensor<1x2x5x8xf16>
}

// -----

func.func @unpaired_past_key_value(
    %ctx: !hip.context,
    %query: tensor<1x2x32xf16>,
    %key: tensor<1x3x16xf16>,
    %value: tensor<1x3x16xf16>,
    %past_key: tensor<1x2x4x8xf16>,
    %seqlens: tensor<1xi32>,
    %total: tensor<i32>,
    %out: tensor<1x2x32xf16>,
    %present_key: tensor<1x2x5x8xf16>,
    %present_value: tensor<1x2x5x8xf16>)
    -> (tensor<1x2x32xf16>, tensor<1x2x5x8xf16>,
        tensor<1x2x5x8xf16>) {
  // expected-error @+1 {{past_key and past_value must both be provided or both be omitted}}
  %result:3 = "hip.gqa"(
      %ctx, %query, %key, %value, %past_key, %seqlens, %total, %out,
      %present_key, %present_value)
      <{num_heads = 4 : i64, kv_num_heads = 2 : i64,
        operandSegmentSizes =
          array<i32: 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>}>
      : (!hip.context, tensor<1x2x32xf16>, tensor<1x3x16xf16>,
         tensor<1x3x16xf16>, tensor<1x2x4x8xf16>, tensor<1xi32>,
         tensor<i32>, tensor<1x2x32xf16>, tensor<1x2x5x8xf16>,
         tensor<1x2x5x8xf16>)
      -> (tensor<1x2x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>)
  return %result#0, %result#1, %result#2 :
      tensor<1x2x32xf16>, tensor<1x2x5x8xf16>, tensor<1x2x5x8xf16>
}
