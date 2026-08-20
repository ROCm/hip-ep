// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-each=0 --test-gqa-reify-failure-atomic %s 2>&1 | FileCheck %s

// CHECK: error: 'hip.gqa' op softcap must be exactly zero
// CHECK: remark: failed GQA reification left IR unchanged
// CHECK-NOT: failed GQA reification mutated IR
func.func @softcap_failure_is_atomic(
    %ctx: !hip.context, %query: tensor<?x?x32xf16>,
    %key: tensor<?x?x16xf16>, %value: tensor<?x?x16xf16>,
    %past_key: tensor<?x2x?x8xf16>, %past_value: tensor<?x2x?x8xf16>,
    %seqlens: tensor<?xi32>, %total: tensor<i32>,
    %out: tensor<?x?x?xf16>, %present_key: tensor<?x2x?x8xf16>,
    %present_value: tensor<?x2x?x8xf16>) -> index {
  %result:3 = hip.gqa(%ctx)
    ins(%query, %key, %value, %past_key, %past_value, %seqlens, %total :
        tensor<?x?x32xf16>, tensor<?x?x16xf16>, tensor<?x?x16xf16>,
        tensor<?x2x?x8xf16>, tensor<?x2x?x8xf16>, tensor<?xi32>, tensor<i32>)
    outs(%out, %present_key, %present_value :
        tensor<?x?x?xf16>, tensor<?x2x?x8xf16>, tensor<?x2x?x8xf16>)
    {num_heads = 4 : i64, kv_num_heads = 2 : i64,
     test.gqa_reify_failure_atomic}
    : tensor<?x?x?xf16>, tensor<?x2x?x8xf16>, tensor<?x2x?x8xf16>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %result#0, %c0 : tensor<?x?x?xf16>
  return %dim : index
}

// CHECK: remark: supported GQA reification succeeded
func.func @supported_int8_per_channel(
    %ctx: !hip.context, %query: tensor<?x?x32xf16>,
    %key: tensor<?x?x16xf16>, %value: tensor<?x?x16xf16>,
    %past_key: tensor<?x2x?x8xi8>, %past_value: tensor<?x2x?x8xi8>,
    %seqlens: tensor<?xi32>, %total: tensor<i32>,
    %k_scale: tensor<16xf32>, %v_scale: tensor<16xf32>,
    %out: tensor<?x?x?xf16>, %present_key: tensor<?x2x?x8xi8>,
    %present_value: tensor<?x2x?x8xi8>)
    -> (tensor<?x?x?xf16>, tensor<?x2x?x8xi8>, tensor<?x2x?x8xi8>) {
  %result:3 = "hip.gqa"(
      %ctx, %query, %key, %value, %past_key, %past_value, %seqlens, %total,
      %k_scale, %v_scale, %out, %present_key, %present_value)
      <{num_heads = 4 : i64, kv_num_heads = 2 : i64,
        k_quant_type = "PER_CHANNEL", v_quant_type = "PER_CHANNEL",
        operandSegmentSizes =
          array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0>}>
      {test.gqa_reify_supported}
      : (!hip.context, tensor<?x?x32xf16>, tensor<?x?x16xf16>,
         tensor<?x?x16xf16>, tensor<?x2x?x8xi8>, tensor<?x2x?x8xi8>,
         tensor<?xi32>, tensor<i32>, tensor<16xf32>, tensor<16xf32>,
         tensor<?x?x?xf16>, tensor<?x2x?x8xi8>, tensor<?x2x?x8xi8>)
      -> (tensor<?x?x?xf16>, tensor<?x2x?x8xi8>, tensor<?x2x?x8xi8>)
  return %result#0, %result#1, %result#2
      : tensor<?x?x?xf16>, tensor<?x2x?x8xi8>, tensor<?x2x?x8xi8>
}

// CHECK: error: 'hip.gqa' op rotary_interleaved must be zero
// CHECK: remark: failed GQA rotary_interleaved reification left IR unchanged
// CHECK: error: 'hip.gqa' op PER_TENSOR KV quantization is unsupported
// CHECK: remark: failed GQA per_tensor reification left IR unchanged
// CHECK: error: 'hip.gqa' op quantized GQA supports only 8-bit KV caches; 4-bit KV caches are unsupported
// CHECK: remark: failed GQA int4 reification left IR unchanged
// CHECK: error: 'hip.gqa' op K/V quantization schemes must match
// CHECK: remark: failed GQA mixed_scheme reification left IR unchanged
// CHECK: error: 'hip.gqa' op quantized GQA past_value element type must be signed int8
// CHECK: remark: failed GQA mixed_dtype reification left IR unchanged
func.func @unsupported_quantization_matrix(
    %ctx: !hip.context, %query: tensor<?x?x32xf16>,
    %key: tensor<?x?x16xf16>, %value: tensor<?x?x16xf16>,
    %past_key: tensor<?x2x?x8xi8>, %past_value: tensor<?x2x?x8xi8>,
    %seqlens: tensor<?xi32>, %total: tensor<i32>,
    %k_scale: tensor<16xf32>, %v_scale: tensor<16xf32>,
    %out: tensor<?x?x?xf16>, %present_key: tensor<?x2x?x8xi8>,
    %present_value: tensor<?x2x?x8xi8>)
    -> (tensor<?x?x?xf16>, tensor<?x2x?x8xi8>, tensor<?x2x?x8xi8>) {
  %result:3 = "hip.gqa"(
      %ctx, %query, %key, %value, %past_key, %past_value, %seqlens, %total,
      %k_scale, %v_scale, %out, %present_key, %present_value)
      <{num_heads = 4 : i64, kv_num_heads = 2 : i64,
        k_quant_type = "PER_CHANNEL", v_quant_type = "PER_CHANNEL",
        operandSegmentSizes =
          array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0>}>
      {test.gqa_reify_quant_matrix}
      : (!hip.context, tensor<?x?x32xf16>, tensor<?x?x16xf16>,
         tensor<?x?x16xf16>, tensor<?x2x?x8xi8>, tensor<?x2x?x8xi8>,
         tensor<?xi32>, tensor<i32>, tensor<16xf32>, tensor<16xf32>,
         tensor<?x?x?xf16>, tensor<?x2x?x8xi8>, tensor<?x2x?x8xi8>)
      -> (tensor<?x?x?xf16>, tensor<?x2x?x8xi8>, tensor<?x2x?x8xi8>)
  return %result#0, %result#1, %result#2
      : tensor<?x?x?xf16>, tensor<?x2x?x8xi8>, tensor<?x2x?x8xi8>
}
