// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify com.microsoft.GroupQueryAttention (via onnx.Custom) is correctly
// lowered to hip.group_query_attention operation in tensor-first mode.
//
// This test validates:
// - Custom ONNX op matching by function_name and domain_name
// - Multi-output op lowering (3 outputs)
// - Attribute preservation (num_heads, kv_num_heads, scale, etc.)
// - f16 element type support
// - Tensor-first DPS: tensor.empty() for each output
//
// Model: Llama-3.1-8B GroupQueryAttention decode step
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(
      %query: tensor<1x1x4096xf16>,
      %key: tensor<1x1x1024xf16>,
      %value: tensor<1x1x1024xf16>,
      %past_key: tensor<1x8x127x128xf16>,
      %past_value: tensor<1x8x127x128xf16>,
      %seqlens_k: tensor<1x1xi32>,
      %total_seq_len: tensor<i32>)
      -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %none1 = "onnx.NoValue"() {value} : () -> none
    %none2 = "onnx.NoValue"() {value} : () -> none

    %out:3 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                            %seqlens_k, %total_seq_len, %none1, %none2)
        <{function_name = "GroupQueryAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 32 : si64,
         kv_num_heads = 8 : si64,
         scale = 0.0883883461 : f32,
         softcap = 0.000000e+00 : f32,
         do_rotary = 0 : si64,
         rotary_interleaved = 0 : si64}
        : (tensor<1x1x4096xf16>, tensor<1x1x1024xf16>, tensor<1x1x1024xf16>,
           tensor<1x8x127x128xf16>, tensor<1x8x127x128xf16>,
           tensor<1x1xi32>, tensor<i32>, none, none)
        -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)

    // Three tensor.empty() inits for the three outputs
    // CHECK: tensor.empty() : tensor<1x1x4096xf16>
    // CHECK: tensor.empty() : tensor<1x8x128x128xf16>
    // CHECK: tensor.empty() : tensor<1x8x128x128xf16>
    // CHECK: hip.group_query_attention
    // CHECK-SAME: kv_num_heads = 8
    // CHECK-SAME: num_heads = 32
    // CHECK-NOT: hip.alloc

    return %out#0, %out#1, %out#2 : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
  }
}
