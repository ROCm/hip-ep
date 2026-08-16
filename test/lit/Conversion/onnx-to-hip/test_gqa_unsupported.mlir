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
