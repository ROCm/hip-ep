// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --split-input-file --mlir-print-ir-after-failure %s 2>&1 | FileCheck %s

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xf16>,
      %past_value: tensor<1x2x4x8xf16>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>, %position_ids: tensor<1x1xi64>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>) {
    %none0 = "onnx.NoValue"() {value} : () -> none
    %none1 = "onnx.NoValue"() {value} : () -> none
    // CHECK: error: GroupQueryAttention position_ids is unsupported by the runtime
    %out:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total,
        %none0, %none1, %position_ids)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xf16>,
           tensor<1x2x4x8xf16>, tensor<1xi32>, tensor<i32>, none, none,
           tensor<1x1xi64>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
            tensor<1x2x5x8xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xf16>,
      %past_value: tensor<1x2x4x8xf16>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>) {
    // CHECK: error: PER_TENSOR KV quantization is unsupported
    %out:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64, k_quant_type = "PER_TENSOR",
         v_quant_type = "PER_TENSOR"}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xf16>,
           tensor<1x2x4x8xf16>, tensor<1xi32>, tensor<i32>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
            tensor<1x2x5x8xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xi8>,
      %past_value: tensor<1x2x4x8xi8>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xi8>,
          tensor<1x2x5x8xi8>) {
    // CHECK: error: quantized GQA supports only 8-bit KV caches; 4-bit KV caches are unsupported
    %out:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64, k_quant_type = "PER_CHANNEL",
         v_quant_type = "PER_CHANNEL", kv_cache_bit_width = 4 : si64}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xi8>,
           tensor<1x2x4x8xi8>, tensor<1xi32>, tensor<i32>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xi8>,
            tensor<1x2x5x8xi8>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x32xf16>, tensor<1x2x5x8xi8>, tensor<1x2x5x8xi8>
  }
}

// -----

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xi8>,
      %past_value: tensor<1x2x4x8xi8>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xi8>,
          tensor<1x2x5x8xi8>) {
    // CHECK: error: K/V quantization schemes must match
    %out:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64, k_quant_type = "PER_CHANNEL",
         v_quant_type = "NONE"}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xi8>,
           tensor<1x2x4x8xi8>, tensor<1xi32>, tensor<i32>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xi8>,
            tensor<1x2x5x8xi8>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x32xf16>, tensor<1x2x5x8xi8>, tensor<1x2x5x8xi8>
  }
}

// -----

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xi8>,
      %past_value: tensor<1x2x4x8xf16>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>, %k_scale: tensor<16xf32>,
      %v_scale: tensor<16xf32>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xi8>,
          tensor<1x2x5x8xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    // CHECK: error: quantized GQA past_value element type must be signed int8
    %out:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total,
        %none, %none, %none, %none, %none, %k_scale, %v_scale)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64, k_quant_type = "PER_CHANNEL",
         v_quant_type = "PER_CHANNEL", kv_cache_bit_width = 8 : si64}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xi8>,
           tensor<1x2x4x8xf16>, tensor<1xi32>, tensor<i32>,
           none, none, none, none, none, tensor<16xf32>, tensor<16xf32>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xi8>,
            tensor<1x2x5x8xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x32xf16>, tensor<1x2x5x8xi8>,
          tensor<1x2x5x8xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xf16>,
      %past_value: tensor<1x2x4x8xf16>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>) {
    // CHECK: error: rotary_interleaved must be zero
    %out:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64, rotary_interleaved = 1 : si64}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xf16>,
           tensor<1x2x4x8xf16>, tensor<1xi32>, tensor<i32>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
            tensor<1x2x5x8xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xf16>,
      %past_value: tensor<1x2x4x8xf16>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>) {
    // CHECK: error: GroupQueryAttention qk_output is unsupported by the runtime
    %out:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64, qk_output = 1 : si64}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xf16>,
           tensor<1x2x4x8xf16>, tensor<1xi32>, tensor<i32>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
            tensor<1x2x5x8xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xf16>,
      %past_value: tensor<1x2x4x8xf16>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>, tensor<1x4x1x5xf16>) {
    // CHECK: error: GroupQueryAttention output_qk is unsupported by the runtime
    %out:4 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xf16>,
           tensor<1x2x4x8xf16>, tensor<1xi32>, tensor<i32>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
            tensor<1x2x5x8xf16>, tensor<1x4x1x5xf16>)
    return %out#0, %out#1, %out#2, %out#3
        : tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>, tensor<1x4x1x5xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xf16>,
      %past_value: tensor<1x2x4x8xf16>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>) {
    // CHECK: error: GroupQueryAttention nonzero softcap is unsupported by the runtime
    %out:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64, softcap = 3.000000e+01 : f32}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xf16>,
           tensor<1x2x4x8xf16>, tensor<1xi32>, tensor<i32>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
            tensor<1x2x5x8xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xf16>,
      %past_value: tensor<1x2x4x8xf16>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>) {
    // CHECK: error: GroupQueryAttention nonzero softcap is unsupported by the runtime
    %out:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64, softcap = 0x7FC00000 : f32}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xf16>,
           tensor<1x2x4x8xf16>, tensor<1xi32>, tensor<i32>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
            tensor<1x2x5x8xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xf16>,
      %past_value: tensor<1x2x4x8xf16>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>) {
    // CHECK: error: GroupQueryAttention nonzero softcap is unsupported by the runtime
    %out:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64, softcap = 0x7F800000 : f32}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xf16>,
           tensor<1x2x4x8xf16>, tensor<1xi32>, tensor<i32>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
            tensor<1x2x5x8xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %query: tensor<1x1x32xf16>, %key: tensor<1x1x16xf16>,
      %value: tensor<1x1x16xf16>, %past_key: tensor<1x2x4x8xf16>,
      %past_value: tensor<1x2x4x8xf16>, %seqlens: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>) {
    // CHECK: error: GroupQueryAttention nonzero softcap is unsupported by the runtime
    %out:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %seqlens, %total)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft", num_heads = 4 : si64,
         kv_num_heads = 2 : si64, softcap = 0x00000001 : f32}
        : (tensor<1x1x32xf16>, tensor<1x1x16xf16>,
           tensor<1x1x16xf16>, tensor<1x2x4x8xf16>,
           tensor<1x2x4x8xf16>, tensor<1xi32>, tensor<i32>)
        -> (tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
            tensor<1x2x5x8xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x32xf16>, tensor<1x2x5x8xf16>,
          tensor<1x2x5x8xf16>
  }
}

// Unsupported forms fail before destination/readback/HIP op creation.
// CHECK-NOT: tensor.empty
// CHECK-NOT: hip.readback
// CHECK-NOT: hip.gqa
