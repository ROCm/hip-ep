// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x640xf16>) -> tensor<1x640xf16> {
    return %arg0 : tensor<1x640xf16>
  }

  func.func @sai_qsa(
      %qk: tensor<?x?x640xf16>,
      %qw: tensor<128xf16>,
      %kw: tensor<128xf16>,
      %cos: tensor<?x?x64xf16>,
      %sin: tensor<?x?x64xf16>,
      %mask: tensor<?x?xi64>,
      %past: tensor<?x?x128xf16>
  ) -> (tensor<?x?x2051xi32>, tensor<?x?x128xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %mask, %past) {
      function_name = "SparseAttentionIndexer",
      domain_name = "com.microsoft",
      policy_mode = "qsa",
      compress_ratio = 4 : si64,
      token_budget = 2048 : si64,
      epsilon = 9.99999997E-7 : f32,
      scale = 0.0883883461 : f32
    } : (tensor<?x?x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<?x?xi64>, tensor<?x?x128xf16>)
      -> (tensor<?x?x2051xi32>, tensor<?x?x128xf16>)
    return %sel, %present : tensor<?x?x2051xi32>, tensor<?x?x128xf16>
  }

  // CHECK-LABEL: func.func @sai_qsa
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[QK:.*]]: tensor<?x?x640xf16>, %[[QW:.*]]: tensor<128xf16>, %[[KW:.*]]: tensor<128xf16>, %[[COS:.*]]: tensor<?x?x64xf16>, %[[SIN:.*]]: tensor<?x?x64xf16>, %[[MASK:.*]]: tensor<?x?xi64>, %[[PAST:.*]]: tensor<?x?x128xf16>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[B:.*]] = tensor.dim %[[QK]], %[[C0]]
  // CHECK: %[[S:.*]] = tensor.dim %[[QK]], %[[C1]]
  // CHECK: %[[SEL:.*]] = tensor.empty(%[[B]], %[[S]]) : tensor<?x?x2051xi32>
  // CHECK: %[[PB:.*]] = tensor.dim %[[QK]], %[[C0]]
  // CHECK: %[[P:.*]] = tensor.dim %[[PAST]], %[[C1]]
  // CHECK: %[[PS:.*]] = tensor.dim %[[QK]], %[[C1]]
  // CHECK: %[[T:.*]] = arith.addi %[[P]], %[[PS]] : index
  // CHECK: %[[PRESENT:.*]] = tensor.empty(%[[PB]], %[[T]]) : tensor<?x?x128xf16>
  // CHECK: %[[R:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<?x?x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<?x?x128xf16>) mask(%[[MASK]] : tensor<?x?xi64>) outs(%[[SEL]], %[[PRESENT]] : tensor<?x?x2051xi32>, tensor<?x?x128xf16>) {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64}
  // CHECK: return %[[R]]#0, %[[R]]#1

  func.func @sai_no_mask_rank2_cache(
      %qk: tensor<2x3x384xf32>,
      %qw: tensor<128xf32>,
      %kw: tensor<128xf32>,
      %cos: tensor<64x32xf32>,
      %sin: tensor<64x32xf32>,
      %past: tensor<2x5x128xf32>
  ) -> (tensor<2x3x67xi32>, tensor<2x8x128xf32>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past, %none, %none, %none, %none, %none, %none) {
      function_name = "SparseAttentionIndexer",
      domain_name = "com.microsoft",
      policy_mode = "qsa",
      compress_ratio = 4 : si64,
      token_budget = 64 : si64
    } : (tensor<2x3x384xf32>, none, tensor<128xf32>, tensor<128xf32>, tensor<64x32xf32>, tensor<64x32xf32>, none, tensor<2x5x128xf32>, none, none, none, none, none, none)
      -> (tensor<2x3x67xi32>, tensor<2x8x128xf32>)
    return %sel, %present : tensor<2x3x67xi32>, tensor<2x8x128xf32>
  }

  // CHECK-LABEL: func.func @sai_no_mask_rank2_cache
  // CHECK: hip.sparse_attention_indexer
  // CHECK-SAME: tensor<64x32xf32>, tensor<64x32xf32>, tensor<2x5x128xf32>) outs(
  // CHECK-SAME: {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 64 : i64}

  func.func @sai_csa_policy(
      %qk: tensor<1x1x640xf16>,
      %qw: tensor<128xf16>,
      %kw: tensor<128xf16>,
      %cos: tensor<1x8x64xf16>,
      %sin: tensor<1x8x64xf16>,
      %mask: tensor<1x8xi64>,
      %past: tensor<1x7x128xf16>
  ) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %mask, %past) {
      function_name = "SparseAttentionIndexer",
      domain_name = "com.microsoft",
      policy_mode = "csa",
      compress_ratio = 4 : si64,
      token_budget = 2048 : si64
    } : (tensor<1x1x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<1x8x64xf16>, tensor<1x8x64xf16>, tensor<1x8xi64>, tensor<1x7x128xf16>)
      -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
    return %sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  }

  // CHECK-LABEL: func.func @sai_csa_policy
  // CHECK-NOT: hip.sparse_attention_indexer
  // CHECK: "onnx.Custom"
  // CHECK-SAME: policy_mode = "csa"

  func.func @sai_separate_key(
      %q: tensor<1x1x512xf16>,
      %k: tensor<1x1x128xf16>,
      %qw: tensor<128xf16>,
      %kw: tensor<128xf16>,
      %cos: tensor<1x8x64xf16>,
      %sin: tensor<1x8x64xf16>,
      %mask: tensor<1x8xi64>,
      %past: tensor<1x7x128xf16>
  ) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
    %sel, %present = "onnx.Custom"(%q, %k, %qw, %kw, %cos, %sin, %mask, %past) {
      function_name = "SparseAttentionIndexer",
      domain_name = "com.microsoft",
      policy_mode = "qsa",
      compress_ratio = 4 : si64,
      token_budget = 2048 : si64
    } : (tensor<1x1x512xf16>, tensor<1x1x128xf16>, tensor<128xf16>, tensor<128xf16>, tensor<1x8x64xf16>, tensor<1x8x64xf16>, tensor<1x8xi64>, tensor<1x7x128xf16>)
      -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
    return %sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  }

  // CHECK-LABEL: func.func @sai_separate_key
  // CHECK-NOT: hip.sparse_attention_indexer
  // CHECK: "onnx.Custom"

  func.func @sai_dynamic_capacity(
      %qk: tensor<?x?x640xf16>,
      %qw: tensor<128xf16>,
      %kw: tensor<128xf16>,
      %cos: tensor<?x?x64xf16>,
      %sin: tensor<?x?x64xf16>,
      %mask: tensor<?x?xi64>,
      %past: tensor<?x?x128xf16>
  ) -> (tensor<?x?x?xi32>, tensor<?x?x128xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %mask, %past) {
      function_name = "SparseAttentionIndexer",
      domain_name = "com.microsoft",
      policy_mode = "qsa",
      compress_ratio = 4 : si64,
      token_budget = 2048 : si64
    } : (tensor<?x?x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<?x?xi64>, tensor<?x?x128xf16>)
      -> (tensor<?x?x?xi32>, tensor<?x?x128xf16>)
    return %sel, %present : tensor<?x?x?xi32>, tensor<?x?x128xf16>
  }

  // CHECK-LABEL: func.func @sai_dynamic_capacity
  // CHECK-NOT: hip.sparse_attention_indexer
  // CHECK: "onnx.Custom"

  func.func @sai_sin_rows_mismatch(
      %qk: tensor<1x1x640xf16>,
      %qw: tensor<128xf16>,
      %kw: tensor<128xf16>,
      %cos: tensor<8x64xf16>,
      %sin: tensor<9x64xf16>,
      %past: tensor<1x7x128xf16>
  ) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past) {
      function_name = "SparseAttentionIndexer",
      domain_name = "com.microsoft",
      policy_mode = "qsa",
      compress_ratio = 4 : si64,
      token_budget = 2048 : si64
    } : (tensor<1x1x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<9x64xf16>, none, tensor<1x7x128xf16>)
      -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
    return %sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  }

  // CHECK-LABEL: func.func @sai_sin_rows_mismatch
  // CHECK-NOT: hip.sparse_attention_indexer
  // CHECK: "onnx.Custom"
}
