// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(
      %query: tensor<4x128xf16>,
      %key: tensor<4x128xf16>,
      %value: tensor<4x128xf16>,
      %key_cache: tensor<8x16x2x16xf16>,
      %value_cache: tensor<8x16x2x16xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>) -> tensor<4x128xf16> {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %0 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                        %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64,
         do_rotary = 0 : si64,
         rotary_interleaved = 0 : si64,
         local_window_size = -1 : si64,
         scale = 0.000000e+00 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<4x128xf16>, tensor<4x128xf16>, tensor<4x128xf16>,
           tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> tensor<4x128xf16>

    // CHECK: tensor.empty() : tensor<4x128xf16>
    // CHECK: hip.paged_attention(%[[CTX]])
    // CHECK-SAME: num_heads = 4
    // CHECK-SAME: kv_num_heads = 2
    // CHECK-NOT: onnx.Custom

    return %0 : tensor<4x128xf16>
  }

  func.func @test_dynamic_packed(
      %query: tensor<?x128xf16>,
      %key_cache: tensor<8x16x2x16xf16>,
      %value_cache: tensor<8x16x2x16xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>) -> tensor<?x128xf16> {

    %none = "onnx.NoValue"() {value} : () -> none

    // CHECK-LABEL: func.func @test_dynamic_packed
    %0 = "onnx.Custom"(%query, %none, %none, %key_cache, %value_cache,
                        %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64}
        : (tensor<?x128xf16>, none, none, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>,
           tensor<1x8xi32>)
        -> tensor<?x128xf16>

    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: tensor.dim {{.*}} : tensor<?x128xf16>
    // CHECK: tensor.empty({{.*}}) : tensor<?x128xf16>
    // CHECK: hip.paged_attention

    return %0 : tensor<?x128xf16>
  }
}
