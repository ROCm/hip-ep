// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(
      %query: tensor<4x64xf16>,
      %key: tensor<4x32xf16>,
      %value: tensor<4x32xf16>,
      %key_cache: tensor<8x16x2x16xf16>,
      %value_cache: tensor<8x16x2x16xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>)
      -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %0:3 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                          %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64,
         do_rotary = 0 : si64,
         local_window_size = -1 : si64,
         scale = 2.500000e-01 : f32,
         softcap = 0.000000e+00 : f32,
         k_quant_type = "NONE",
         k_cache_dtype = "\00",
         v_cache_dtype = "\00",
         kv_cache_layout = "SEPARATE"}
        : (tensor<4x64xf16>, tensor<4x32xf16>, tensor<4x32xf16>,
           tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>)

    // The present caches are fresh outputs, not the past cache values.
    // CHECK: tensor.empty() : tensor<4x64xf16>
    // CHECK: %[[KCO:.*]] = tensor.empty() : tensor<8x16x2x16xf16>
    // CHECK: %[[VCO:.*]] = tensor.empty() : tensor<8x16x2x16xf16>
    // CHECK: hip.paged_attention(%[[CTX]])
    // CHECK-SAME: key(
    // CHECK-SAME: value(
    // CHECK-SAME: outs(%{{.*}}, %[[KCO]], %[[VCO]] :
    // Attributes printed in alphabetical order by MLIR's attr-dict printer.
    // CHECK-SAME: kv_num_heads = 2
    // CHECK-SAME: num_heads = 4
    // CHECK-SAME: scale = 2.500000e-01
    // CHECK-NOT: onnx.Custom

    return %0#0, %0#1, %0#2
        : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  }

  func.func @test_attention_metadata(
      %query: tensor<?x64xf16>,
      %key: tensor<?x32xf16>,
      %value: tensor<?x32xf16>,
      %key_cache: tensor<8x16x2x16xf16>,
      %value_cache: tensor<8x16x2x16xf16>,
      %cum_seq: tensor<?xi32>,
      %past_seqlens: tensor<?xi32>,
      %block_table: tensor<?x8xi32>,
      %metadata: tensor<3xi32>)
      -> (tensor<?x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {

    %none = "onnx.NoValue"() {value} : () -> none

    // CHECK-LABEL: func.func @test_attention_metadata
    %0:3 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                          %cum_seq, %past_seqlens, %block_table,
                          %none, %none, %none, %none, %none, %none, %none,
                          %none, %metadata)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64}
        : (tensor<?x64xf16>, tensor<?x32xf16>, tensor<?x32xf16>,
           tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>, tensor<?xi32>,
           tensor<?xi32>, tensor<?x8xi32>, none, none, none, none, none,
           none, none, none, tensor<3xi32>)
        -> (tensor<?x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>)

    // CHECK: tensor.empty(%{{.*}}) : tensor<?x64xf16>
    // CHECK: %[[KCO:.*]] = tensor.empty() : tensor<8x16x2x16xf16>
    // CHECK: %[[VCO:.*]] = tensor.empty() : tensor<8x16x2x16xf16>
    // CHECK: hip.paged_attention(%{{.*}}) ins({{.*}}) key({{.*}}) value({{.*}}) metadata(%{{.*}} : tensor<3xi32>) outs(%{{.*}}, %[[KCO]], %[[VCO]] :
    // CHECK-NOT: onnx.Custom

    return %0#0, %0#1, %0#2
        : tensor<?x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  }

  func.func @test_dynamic_packed(
      %query: tensor<?x128xf16>,
      %key_cache: tensor<8x16x2x16xf16>,
      %value_cache: tensor<8x16x2x16xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>)
      -> (tensor<?x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {

    %none = "onnx.NoValue"() {value} : () -> none

    // CHECK-LABEL: func.func @test_dynamic_packed
    %0:3 = "onnx.Custom"(%query, %none, %none, %key_cache, %value_cache,
                          %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64}
        : (tensor<?x128xf16>, none, none, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>,
           tensor<1x8xi32>)
        -> (tensor<?x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>)

    // Absent key and value stay absent on the HIP op.
    // CHECK: hip.paged_attention(%{{.*}}) ins({{.*}}) outs(
    // CHECK-NOT: key(
    // CHECK-NOT: onnx.Custom

    return %0#0, %0#1, %0#2
        : tensor<?x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  }

  func.func @test_rotary_passthrough(
      %query: tensor<4x64xf16>,
      %key: tensor<4x32xf16>,
      %value: tensor<4x32xf16>,
      %key_cache: tensor<8x16x2x16xf16>,
      %value_cache: tensor<8x16x2x16xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>,
      %cos: tensor<16x8xf16>,
      %sin: tensor<16x8xf16>)
      -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {

    // CHECK-LABEL: func.func @test_rotary_passthrough
    %0:3 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                          %cum_seq, %past_seqlens, %block_table, %cos, %sin)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64,
         do_rotary = 1 : si64,
         rotary_interleaved = 1 : si64}
        : (tensor<4x64xf16>, tensor<4x32xf16>, tensor<4x32xf16>,
           tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>, tensor<16x8xf16>,
           tensor<16x8xf16>)
        -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>)

    // The rotary tables and attributes reach the HIP op unchanged.
    // CHECK: hip.paged_attention(%{{.*}}) ins({{.*}}) key({{.*}}) value({{.*}}) cos(%{{.*}} : tensor<16x8xf16>) sin(%{{.*}} : tensor<16x8xf16>) outs(
    // CHECK-SAME: do_rotary = 1
    // CHECK-SAME: rotary_interleaved = 1
    // CHECK-NOT: onnx.Custom

    return %0#0, %0#1, %0#2
        : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  }

  func.func @test_local_window_passthrough(
      %query: tensor<4x64xf16>,
      %key: tensor<4x32xf16>,
      %value: tensor<4x32xf16>,
      %key_cache: tensor<8x16x2x16xf16>,
      %value_cache: tensor<8x16x2x16xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>)
      -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {

    // CHECK-LABEL: func.func @test_local_window_passthrough
    %0:3 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                          %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64,
         local_window_size = 8 : si64}
        : (tensor<4x64xf16>, tensor<4x32xf16>, tensor<4x32xf16>,
           tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>)

    // CHECK: hip.paged_attention(%{{.*}})
    // CHECK-SAME: local_window_size = 8
    // CHECK-NOT: onnx.Custom

    return %0#0, %0#1, %0#2
        : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  }

  // A node without the cache outputs is outside the supported subset.
  func.func @test_unsupported_no_cache_outputs(
      %query: tensor<4x64xf16>,
      %key: tensor<4x32xf16>,
      %value: tensor<4x32xf16>,
      %key_cache: tensor<8x16x2x16xf16>,
      %value_cache: tensor<8x16x2x16xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>) -> tensor<4x64xf16> {

    // CHECK-LABEL: func.func @test_unsupported_no_cache_outputs
    %0 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                        %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64}
        : (tensor<4x64xf16>, tensor<4x32xf16>, tensor<4x32xf16>,
           tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> tensor<4x64xf16>

    // CHECK: onnx.Custom
    // CHECK-NOT: hip.paged_attention

    return %0 : tensor<4x64xf16>
  }

  func.func @test_unsupported_softcap(
      %query: tensor<4x64xf16>,
      %key: tensor<4x32xf16>,
      %value: tensor<4x32xf16>,
      %key_cache: tensor<8x16x2x16xf16>,
      %value_cache: tensor<8x16x2x16xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>)
      -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {

    // CHECK-LABEL: func.func @test_unsupported_softcap
    %0:3 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                          %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64,
         softcap = 1.000000e+01 : f32}
        : (tensor<4x64xf16>, tensor<4x32xf16>, tensor<4x32xf16>,
           tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>)

    // CHECK: onnx.Custom
    // CHECK-NOT: hip.paged_attention

    return %0#0, %0#1, %0#2
        : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  }

  func.func @test_unsupported_non_causal(
      %query: tensor<4x64xf16>,
      %key: tensor<4x32xf16>,
      %value: tensor<4x32xf16>,
      %key_cache: tensor<8x16x2x16xf16>,
      %value_cache: tensor<8x16x2x16xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>)
      -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {

    // CHECK-LABEL: func.func @test_unsupported_non_causal
    %0:3 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                          %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64,
         is_causal = 0 : si64}
        : (tensor<4x64xf16>, tensor<4x32xf16>, tensor<4x32xf16>,
           tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>)

    // CHECK: onnx.Custom
    // CHECK-NOT: hip.paged_attention

    return %0#0, %0#1, %0#2
        : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  }

  func.func @test_unsupported_quantized_cache(
      %query: tensor<4x64xf16>,
      %key: tensor<4x32xf16>,
      %value: tensor<4x32xf16>,
      %key_cache: tensor<8x16x2x16xf16>,
      %value_cache: tensor<8x16x2x16xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>)
      -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {

    // CHECK-LABEL: func.func @test_unsupported_quantized_cache
    %0:3 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                          %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64,
         k_quant_type = "PER_TENSOR"}
        : (tensor<4x64xf16>, tensor<4x32xf16>, tensor<4x32xf16>,
           tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>)

    // CHECK: onnx.Custom
    // CHECK-NOT: hip.paged_attention

    return %0#0, %0#1, %0#2
        : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  }

  func.func @test_unsupported_bf16(
      %query: tensor<4x64xbf16>,
      %key: tensor<4x32xbf16>,
      %value: tensor<4x32xbf16>,
      %key_cache: tensor<8x16x2x16xbf16>,
      %value_cache: tensor<8x16x2x16xbf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>)
      -> (tensor<4x64xbf16>, tensor<8x16x2x16xbf16>, tensor<8x16x2x16xbf16>) {

    // CHECK-LABEL: func.func @test_unsupported_bf16
    %0:3 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                          %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64}
        : (tensor<4x64xbf16>, tensor<4x32xbf16>, tensor<4x32xbf16>,
           tensor<8x16x2x16xbf16>, tensor<8x16x2x16xbf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> (tensor<4x64xbf16>, tensor<8x16x2x16xbf16>, tensor<8x16x2x16xbf16>)

    // Only f16 is supported. The node stays unconverted.
    // CHECK: onnx.Custom
    // CHECK-NOT: hip.paged_attention

    return %0#0, %0#1, %0#2
        : tensor<4x64xbf16>, tensor<8x16x2x16xbf16>, tensor<8x16x2x16xbf16>
  }
}
