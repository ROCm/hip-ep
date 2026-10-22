// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x6144xf16>) -> tensor<1x6144xf16> {
    return %arg0 : tensor<1x6144xf16>
  }

  func.func @dsa_selected_only(
      %q: tensor<?x?x6144xf16>,
      %k: tensor<?x?x512xf16>,
      %v: tensor<?x?x512xf16>,
      %past_k: tensor<?x2x?x256xf16>,
      %past_v: tensor<?x2x?x256xf16>,
      %idx: tensor<?x2051xi32>,
      %cnt: tensor<?xi32>,
      %seqlens: tensor<?xi32>,
      %total: tensor<i32>
  ) -> (tensor<?x?x6144xf16>, tensor<?x2x?x256xf16>, tensor<?x2x?x256xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total, %none, %none, %none, %none, %none, %none) {
      function_name = "DynamicSparseAttention",
      domain_name = "com.microsoft",
      num_heads = 24 : si64,
      kv_num_heads = 2 : si64,
      scale = 6.250000e-02 : f32,
      is_causal = 1 : si64,
      attention_mode = "selected_only",
      selected_kv_source = "main",
      do_rotary = 0 : si64,
      rotary_interleaved = 0 : si64,
      qk_norm_epsilon = 9.99999997E-7 : f32
    } : (tensor<?x?x6144xf16>, tensor<?x?x512xf16>, tensor<?x?x512xf16>, tensor<?x2x?x256xf16>, tensor<?x2x?x256xf16>, none, none, tensor<?x2051xi32>, tensor<?xi32>, tensor<?xi32>, tensor<i32>, none, none, none, none, none, none)
      -> (tensor<?x?x6144xf16>, tensor<?x2x?x256xf16>, tensor<?x2x?x256xf16>)
    return %out, %pk, %pv : tensor<?x?x6144xf16>, tensor<?x2x?x256xf16>, tensor<?x2x?x256xf16>
  }

  // CHECK-LABEL: func.func @dsa_selected_only
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[Q:.*]]: tensor<?x?x6144xf16>, %[[K:.*]]: tensor<?x?x512xf16>, %[[V:.*]]: tensor<?x?x512xf16>, %[[PK:.*]]: tensor<?x2x?x256xf16>, %[[PV:.*]]: tensor<?x2x?x256xf16>, %[[IDX:.*]]: tensor<?x2051xi32>, %[[CNT:.*]]: tensor<?xi32>, %[[SEQ:.*]]: tensor<?xi32>, %{{.*}}: tensor<i32>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[B:.*]] = tensor.dim %[[Q]], %[[C0]]
  // CHECK: %[[S:.*]] = tensor.dim %[[Q]], %[[C1]]
  // CHECK: %[[OUT:.*]] = tensor.empty(%[[B]], %[[S]]) : tensor<?x?x6144xf16>
  // CHECK: %[[KB:.*]] = tensor.dim %[[PK]], %[[C0]]
  // CHECK: %[[KCAP:.*]] = tensor.dim %[[PK]], %[[C2]]
  // CHECK: %[[PKI:.*]] = tensor.empty(%[[KB]], %[[KCAP]]) : tensor<?x2x?x256xf16>
  // CHECK: %[[VB:.*]] = tensor.dim %[[PV]], %[[C0]]
  // CHECK: %[[VCAP:.*]] = tensor.dim %[[PV]], %[[C2]]
  // CHECK: %[[PVI:.*]] = tensor.empty(%[[VB]], %[[VCAP]]) : tensor<?x2x?x256xf16>
  // CHECK: %[[R:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[Q]], %[[K]], %[[V]], %[[PK]], %[[PV]], %[[IDX]], %[[CNT]], %[[SEQ]] : tensor<?x?x6144xf16>, tensor<?x?x512xf16>, tensor<?x?x512xf16>, tensor<?x2x?x256xf16>, tensor<?x2x?x256xf16>, tensor<?x2051xi32>, tensor<?xi32>, tensor<?xi32>) outs(%[[OUT]], %[[PKI]], %[[PVI]] : tensor<?x?x6144xf16>, tensor<?x2x?x256xf16>, tensor<?x2x?x256xf16>) {kv_num_heads = 2 : i64, num_heads = 24 : i64, scale = 6.250000e-02 : f32}
  // CHECK: return %[[R]]#0, %[[R]]#1, %[[R]]#2

  func.func @dsa_default_scale(
      %q: tensor<1x4x1024xf16>,
      %k: tensor<1x4x128xf16>,
      %v: tensor<1x4x128xf16>,
      %past_k: tensor<1x1x64x128xf16>,
      %past_v: tensor<1x1x64x128xf16>,
      %idx: tensor<4x16xi32>,
      %cnt: tensor<4xi32>,
      %seqlens: tensor<1xi32>,
      %total: tensor<i32>
  ) -> (tensor<1x4x1024xf16>, tensor<1x1x64x128xf16>, tensor<1x1x64x128xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total) {
      function_name = "DynamicSparseAttention",
      domain_name = "com.microsoft",
      num_heads = 8 : si64,
      kv_num_heads = 1 : si64
    } : (tensor<1x4x1024xf16>, tensor<1x4x128xf16>, tensor<1x4x128xf16>, tensor<1x1x64x128xf16>, tensor<1x1x64x128xf16>, none, none, tensor<4x16xi32>, tensor<4xi32>, tensor<1xi32>, tensor<i32>)
      -> (tensor<1x4x1024xf16>, tensor<1x1x64x128xf16>, tensor<1x1x64x128xf16>)
    return %out, %pk, %pv : tensor<1x4x1024xf16>, tensor<1x1x64x128xf16>, tensor<1x1x64x128xf16>
  }

  // CHECK-LABEL: func.func @dsa_default_scale
  // CHECK: hip.dynamic_sparse_attention
  // CHECK-SAME: {kv_num_heads = 1 : i64, num_heads = 8 : i64, scale = 0.0883883461 : f32}

  func.func @dsa_local_plus_selected(
      %q: tensor<1x1x6144xf16>,
      %k: tensor<1x1x512xf16>,
      %v: tensor<1x1x512xf16>,
      %past_k: tensor<1x2x4096x256xf16>,
      %past_v: tensor<1x2x4096x256xf16>,
      %idx: tensor<1x2051xi32>,
      %cnt: tensor<1xi32>,
      %seqlens: tensor<1xi32>,
      %total: tensor<i32>
  ) -> (tensor<1x1x6144xf16>, tensor<1x2x4096x256xf16>, tensor<1x2x4096x256xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total, %none, %none, %none, %none, %none, %none) {
      function_name = "DynamicSparseAttention",
      domain_name = "com.microsoft",
      num_heads = 24 : si64,
      kv_num_heads = 2 : si64,
      attention_mode = "local_plus_selected",
      local_window_size = 128 : si64
    } : (tensor<1x1x6144xf16>, tensor<1x1x512xf16>, tensor<1x1x512xf16>, tensor<1x2x4096x256xf16>, tensor<1x2x4096x256xf16>, none, none, tensor<1x2051xi32>, tensor<1xi32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, none)
      -> (tensor<1x1x6144xf16>, tensor<1x2x4096x256xf16>, tensor<1x2x4096x256xf16>)
    return %out, %pk, %pv : tensor<1x1x6144xf16>, tensor<1x2x4096x256xf16>, tensor<1x2x4096x256xf16>
  }

  // CHECK-LABEL: func.func @dsa_local_plus_selected
  // CHECK-NOT: hip.dynamic_sparse_attention
  // CHECK: "onnx.Custom"
  // CHECK-SAME: attention_mode = "local_plus_selected"

  func.func @dsa_head_sink(
      %q: tensor<1x1x6144xf16>,
      %k: tensor<1x1x512xf16>,
      %v: tensor<1x1x512xf16>,
      %past_k: tensor<1x2x4096x256xf16>,
      %past_v: tensor<1x2x4096x256xf16>,
      %idx: tensor<1x2051xi32>,
      %cnt: tensor<1xi32>,
      %seqlens: tensor<1xi32>,
      %total: tensor<i32>,
      %sink: tensor<24xf16>
  ) -> (tensor<1x1x6144xf16>, tensor<1x2x4096x256xf16>, tensor<1x2x4096x256xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %idx, %cnt, %seqlens, %total, %none, %none, %none, %none, %none, %sink) {
      function_name = "DynamicSparseAttention",
      domain_name = "com.microsoft",
      num_heads = 24 : si64,
      kv_num_heads = 2 : si64
    } : (tensor<1x1x6144xf16>, tensor<1x1x512xf16>, tensor<1x1x512xf16>, tensor<1x2x4096x256xf16>, tensor<1x2x4096x256xf16>, none, none, tensor<1x2051xi32>, tensor<1xi32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, tensor<24xf16>)
      -> (tensor<1x1x6144xf16>, tensor<1x2x4096x256xf16>, tensor<1x2x4096x256xf16>)
    return %out, %pk, %pv : tensor<1x1x6144xf16>, tensor<1x2x4096x256xf16>, tensor<1x2x4096x256xf16>
  }

  // CHECK-LABEL: func.func @dsa_head_sink
  // CHECK-NOT: hip.dynamic_sparse_attention
  // CHECK: "onnx.Custom"
}
