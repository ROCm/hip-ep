// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Regression guard for the output-allocator KV-cache invariant.
//
// In allocator mode the EP callback hands the DLL's in-graph output shape to
// GetOutput verbatim (no override). That is sound only because each dynamic-seq
// present.* output is sized in-graph from its matching past input buffer's
// actual extent -- under OGA past_present_share_buffer that is the max_length
// capacity buffer, so GetOutput returns the pre-bound shared OrtValue and the
// past==present pointer identity is preserved.
//
// This test pins that invariant: each dynamic present.* output must lower to a
// hip.alloc_output whose dynamic operand is a memref.dim of the matching
// past_key_values input.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-to-hip-pipeline=use-output-allocator=true 2>&1 | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.use_output_allocator = true
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:    %[[PK:[a-zA-Z0-9_]+]]: memref<1x8x?x128xf16> {onnx.name = "past_key_values.0.key"}
// CHECK-SAME:    %[[PV:[a-zA-Z0-9_]+]]: memref<1x8x?x128xf16> {onnx.name = "past_key_values.0.value"}
// present.0.key: dynamic dim comes from memref.dim of past_key (NOT attention_mask).
// CHECK:         %[[DK:.*]] = memref.dim %[[PK]], %{{.*}} : memref<1x8x?x128xf16>
// CHECK:         hip.alloc_output(%{{.*}}, %[[DK]]) {out_idx = 0 : i64} : memref<1x8x?x128xf16>
// present.0.value: dynamic dim comes from memref.dim of past_value.
// CHECK:         %[[DV:.*]] = memref.dim %[[PV]], %{{.*}} : memref<1x8x?x128xf16>
// CHECK:         hip.alloc_output(%{{.*}}, %[[DV]]) {out_idx = 1 : i64} : memref<1x8x?x128xf16>

module {
  func.func @main_graph(
      %q: tensor<1x1x4096xf16> {onnx.name = "q"},
      %k: tensor<1x1x1024xf16> {onnx.name = "k"},
      %v: tensor<1x1x1024xf16> {onnx.name = "v"},
      %past_key: tensor<1x8x?x128xf16> {onnx.name = "past_key_values.0.key"},
      %past_value: tensor<1x8x?x128xf16> {onnx.name = "past_key_values.0.value"},
      %seqlens: tensor<1x1xi32> {onnx.name = "seqlens_k_in"},
      %total: tensor<i32> {onnx.name = "total_seq"})
      -> (tensor<1x8x?x128xf16> {onnx.name = "present.0.key"},
          tensor<1x8x?x128xf16> {onnx.name = "present.0.value"}) {
    %none0 = "onnx.NoValue"() {value} : () -> none
    %none1 = "onnx.NoValue"() {value} : () -> none
    %r:3 = "onnx.Custom"(%q, %k, %v, %past_key, %past_value, %seqlens, %total, %none0, %none1)
      {function_name = "GroupQueryAttention", do_rotary = 0 : si64, domain_name = "com.microsoft",
       kv_num_heads = 8 : si64, num_heads = 32 : si64, rotary_interleaved = 0 : si64,
       scale = 0.0883883461 : f32, softcap = 0.000000e+00 : f32}
      : (tensor<1x1x4096xf16>, tensor<1x1x1024xf16>, tensor<1x1x1024xf16>,
         tensor<1x8x?x128xf16>, tensor<1x8x?x128xf16>, tensor<1x1xi32>, tensor<i32>, none, none)
      -> (tensor<1x1x4096xf16>, tensor<1x8x?x128xf16>, tensor<1x8x?x128xf16>)
    "onnx.Return"(%r#1, %r#2) : (tensor<1x8x?x128xf16>, tensor<1x8x?x128xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
