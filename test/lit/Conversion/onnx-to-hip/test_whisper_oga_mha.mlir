// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Whisper's decoder as onnxruntime-genai exports it (no surgery):
//   * cross-attention K/V are function arguments with a dynamic batch,
//     [?, N, Skv, d]: lowered to hip.gqa(no_causal) with seqlens_k = Skv
//     splatted to the batch ONCE at the function entry and shared by every
//     cross-attention layer (the GQA runtime caches seqlens_k by address);
//   * self-attention takes past_key/past_value [?, N, P, d] unshared with
//     present = concat(past, this step), no past_sequence_length: lowered
//     to causal hip.gqa with seqlens_k = P + S - 1, total P + S, and present
//     buffers of P + S rows.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(
      %q:  tensor<?x?x384xf16>,
      %k:  tensor<?x?x384xf16>,
      %v:  tensor<?x?x384xf16>,
      %pk: tensor<?x6x?x64xf16>,
      %pv: tensor<?x6x?x64xf16>,
      %ck0: tensor<?x6x1500x64xf16>,
      %cv0: tensor<?x6x1500x64xf16>,
      %ck1: tensor<?x6x1500x64xf16>,
      %cv1: tensor<?x6x1500x64xf16>)
      -> (tensor<?x?x384xf16>, tensor<?x6x?x64xf16>, tensor<?x6x?x64xf16>,
          tensor<?x?x384xf16>, tensor<?x?x384xf16>) {

    // CHECK-LABEL: func.func @main_graph
    // The shared cross-attention seqlens_k comes first in the body.
    // CHECK: %[[SKV:.*]] = arith.constant 1500 : i32
    // CHECK: %[[B:.*]] = tensor.dim
    // CHECK: %[[CROSS:.*]] = tensor.splat %[[SKV]][%[[B]]] {{.*}} : tensor<?xi32>
    // CHECK-NOT: hip.cross_seqlens

    %n = "onnx.NoValue"() {value} : () -> none

    // Self-attention: seqlens_k = P + S - 1, causal, present P + S.
    // CHECK: %[[P:.*]] = tensor.dim %{{.*}}, %c2
    // CHECK: %[[S:.*]] = tensor.dim %{{.*}}, %c1
    // CHECK: %[[T:.*]] = arith.addi %[[P]], %[[S]]
    // CHECK: tensor.empty(%{{.*}}, %[[T]]) : tensor<?x6x?x64xf16>
    // CHECK: tensor.empty(%{{.*}}, %[[T]]) : tensor<?x6x?x64xf16>
    // Positional operands: query, key, value, past_key, past_value, seqlens_k, total.
    // CHECK: hip.gqa(%{{.*}}) ins(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}} : {{.*}})
    // CHECK-SAME: {kv_num_heads = 6 : i64, num_heads = 6 : i64
    %o, %pk2, %pv2 = "onnx.Custom"(%q, %k, %v, %n, %n, %n, %pk, %pv)
         <{function_name = "MultiHeadAttention"}>
         {domain_name = "com.microsoft", num_heads = 6 : si64,
          scale = 1.250000e-01 : f32, unidirectional = 1 : si64}
         : (tensor<?x?x384xf16>, tensor<?x?x384xf16>, tensor<?x?x384xf16>,
            none, none, none, tensor<?x6x?x64xf16>, tensor<?x6x?x64xf16>)
         -> (tensor<?x?x384xf16>, tensor<?x6x?x64xf16>, tensor<?x6x?x64xf16>)

    // Two cross-attention layers share the one seqlens_k.
    // CHECK: hip.gqa(%{{.*}}) ins(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %[[CROSS]], %{{[^,]+}} : {{.*}})
    // CHECK-SAME: no_causal = true
    // CHECK: hip.gqa(%{{.*}}) ins(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %[[CROSS]], %{{[^,]+}} : {{.*}})
    // CHECK-SAME: no_causal = true
    // CHECK-NOT: hip.multi_head_attention
    %c0 = "onnx.Custom"(%o, %ck0, %cv0, %n, %n, %n, %n, %n)
         <{function_name = "MultiHeadAttention"}>
         {domain_name = "com.microsoft", num_heads = 6 : si64,
          scale = 1.250000e-01 : f32, unidirectional = 0 : si64}
         : (tensor<?x?x384xf16>, tensor<?x6x1500x64xf16>, tensor<?x6x1500x64xf16>,
            none, none, none, none, none) -> tensor<?x?x384xf16>
    %c1 = "onnx.Custom"(%c0, %ck1, %cv1, %n, %n, %n, %n, %n)
         <{function_name = "MultiHeadAttention"}>
         {domain_name = "com.microsoft", num_heads = 6 : si64,
          scale = 1.250000e-01 : f32, unidirectional = 0 : si64}
         : (tensor<?x?x384xf16>, tensor<?x6x1500x64xf16>, tensor<?x6x1500x64xf16>,
            none, none, none, none, none) -> tensor<?x?x384xf16>
    return %o, %pk2, %pv2, %c0, %c1
        : tensor<?x?x384xf16>, tensor<?x6x?x64xf16>, tensor<?x6x?x64xf16>,
          tensor<?x?x384xf16>, tensor<?x?x384xf16>
  }
}
