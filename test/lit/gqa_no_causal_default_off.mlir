// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Regression guard for the new `no_causal` attribute on hip.gqa:
// the default lowering of com.microsoft.GroupQueryAttention MUST emit
// `no_causal = false` (or no attribute, which is equivalent given the
// DefaultValuedAttr<BoolAttr, "false"> definition).
//
// This protects every existing Llama / gpt-oss path against an accidental
// flip to no_causal=true (which would skip the causal mask and silently
// corrupt prefill / multi-token sliding-window decodes).
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

    // hip.gqa with default no_causal must NOT emit no_causal=true. Note: MLIR
    // omits DefaultValuedAttr<BoolAttr,"false"> entirely when at the default
    // value, so the regression guard is the negative check below: any future
    // accidental flip of the default to true would render `no_causal = true`
    // explicitly in the assembly and the test would fail.
    // CHECK: hip.gqa
    // CHECK-NOT: no_causal = true

    return %out#0, %out#1, %out#2 : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
  }
}
