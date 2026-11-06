// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Output-allocator KV-cache invariant for native ONNX Attention.
//
// Native ONNX Attention uses the full additive mask key extent as its logical
// KV length and preserves a larger preallocated past extent for the present
// output, so a shared past/present buffer keeps its capacity.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-to-hip-pipeline 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:    %[[MASK:[a-zA-Z0-9_]+]]: memref<1x1x?x?xf16> {onnx.name = "attention_bias"}
// CHECK-SAME:    %[[APK:[a-zA-Z0-9_]+]]: memref<1x8x?x128xf16> {onnx.name = "past_key_values.0.key"}
// CHECK:         %[[PAST:.*]] = memref.dim %[[APK]], %{{.*}} : memref<1x8x?x128xf16>
// CHECK:         %[[LOGICAL:.*]] = memref.dim %[[MASK]], %{{.*}} : memref<1x1x?x?xf16>
// CHECK:         %[[CAPACITY:.*]] = arith.maxsi %[[PAST]], %[[LOGICAL]]
// CHECK:         hip.alloc_output(%{{.*}}, %[[CAPACITY]]) {out_idx = 0 : i64} : memref<1x8x?x128xf16>

module {
  func.func @main_graph(
      %q: tensor<1x?x4096xf16> {onnx.name = "q"},
      %k: tensor<1x?x1024xf16> {onnx.name = "k"},
      %v: tensor<1x?x1024xf16> {onnx.name = "v"},
      %mask: tensor<1x1x?x?xf16> {onnx.name = "attention_bias"},
      %past_key: tensor<1x8x?x128xf16> {onnx.name = "past_key_values.0.key"},
      %past_value: tensor<1x8x?x128xf16> {onnx.name = "past_key_values.0.value"})
      -> (tensor<1x8x?x128xf16> {onnx.name = "present.0.key"},
          tensor<1x8x?x128xf16> {onnx.name = "present.0.value"}) {
    %r:3 = "onnx.Attention"(%q, %k, %v, %mask, %past_key, %past_value)
      {is_causal = 1 : si64, kv_num_heads = 8 : si64,
       q_num_heads = 32 : si64, scale = 0.0883883461 : f32,
       softcap = 0.000000e+00 : f32}
      : (tensor<1x?x4096xf16>, tensor<1x?x1024xf16>,
         tensor<1x?x1024xf16>, tensor<1x1x?x?xf16>,
         tensor<1x8x?x128xf16>, tensor<1x8x?x128xf16>)
      -> (tensor<1x?x4096xf16>, tensor<1x8x?x128xf16>,
          tensor<1x8x?x128xf16>)
    "onnx.Return"(%r#1, %r#2)
      : (tensor<1x8x?x128xf16>, tensor<1x8x?x128xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
