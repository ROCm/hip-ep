// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

// The conversion requires a @main_graph entry function.
func.func @main_graph(%arg0: tensor<1xf16>) -> tensor<1xf16> {
  return %arg0 : tensor<1xf16>
}

// The selection and its counts only feed the attention, so the two ops pair:
// the indexer gets pair_format and the attention its past_sequence_length.
// CHECK-LABEL: func.func @pair(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x2x640xf16>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x8x128xf16>,
// CHECK-SAME:      %[[PSL:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[Q:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[K:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[V:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<1x8x128xf16>) {
// CHECK-NEXT:      %[[QW:.*]] = hip.constant {serialization_order = 4 : i64, value = dense<1.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[KW:.*]] = hip.constant {serialization_order = 5 : i64, value = dense<2.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x2x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[COUNTS:.*]] = tensor.empty() : tensor<1x2xi32>
// CHECK-NEXT:      %[[SAI:.*]]:3 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<1x2x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x8x128xf16>)
// CHECK-SAME:          past_sequence_length(%[[PSL]] : tensor<1xi32>)
// CHECK-SAME:          outs(%[[SEL]], %[[PRESENT]] : tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          counts(%[[COUNTS]] : tensor<1x2xi32>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, pair_format, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x2x2051xi32>, tensor<1x8x128xf16>, tensor<1x2xi32>
// CHECK-NEXT:      %[[SEL_ROWS:.*]] = tensor.collapse_shape %[[SAI]]#0 {{\[\[}}0, 1], [2]] : tensor<1x2x2051xi32> into tensor<2x2051xi32>
// CHECK-NEXT:      %[[COUNT_ROWS:.*]] = tensor.collapse_shape %[[SAI]]#2 {{\[\[}}0, 1]] : tensor<1x2xi32> into tensor<2xi32>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[DSA:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[Q]] : tensor<1x2x512xf16>) kv(%[[K]], %[[V]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[SEL_ROWS]], %[[COUNT_ROWS]], %[[SEQLENS]] : tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          pair(%[[PSL]] : tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      return %[[DSA]]#0, %[[SAI]]#1 : tensor<1x2x512xf16>, tensor<1x8x128xf16>
// CHECK-NEXT:    }
func.func @pair(
    %qk: tensor<1x2x640xf16>, %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>,
    %past: tensor<1x8x128xf16>, %psl: tensor<1xi32>,
    %q: tensor<1x2x512xf16>, %k: tensor<1x2x128xf16>, %v: tensor<1x2x128xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %seqlens: tensor<1xi32>, %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<1x8x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %last = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %rows = "onnx.Constant"() {value = dense<[-1, 2051]> : tensor<2xi64>} : () -> tensor<2xi64>
  %qw = "onnx.Constant"() {value = dense<1.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %kw = "onnx.Constant"() {value = dense<2.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past, %none, %none, %none, %none, %psl) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x2x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, none, tensor<1x8x128xf16>, none, none, none, none, tensor<1xi32>)
    -> (tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<1x2x2051xi32>, tensor<i32>) -> tensor<1x2x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<1x2x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<1x2x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %last) {keepdims = 0 : si64} : (tensor<1x2x2051xi32>, tensor<1xi64>) -> tensor<1x2xi32>
  %sel_r = "onnx.Reshape"(%sel, %rows) {allowzero = 0 : si64} : (tensor<1x2x2051xi32>, tensor<2xi64>) -> tensor<2x2051xi32>
  %cnt_r = "onnx.Reshape"(%counts, %last) {allowzero = 0 : si64} : (tensor<1x2xi32>, tensor<1xi64>) -> tensor<2xi32>
  %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %sel_r, %cnt_r, %seqlens, %total, %none, %none, %none, %none, %none, %none) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32,
    attention_mode = "selected_only", selected_kv_source = "main"
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, none)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  return %out, %present : tensor<1x2x512xf16>, tensor<1x8x128xf16>
}

// The selection is also a graph output, so it keeps its score order.
// CHECK-LABEL: func.func @selection_output(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x2x640xf16>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x8x128xf16>,
// CHECK-SAME:      %[[PSL:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[Q:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[K:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[V:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<1x2x2051xi32>) {
// CHECK-NEXT:      %[[QW:.*]] = hip.constant {serialization_order = 10 : i64, value = dense<1.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[KW:.*]] = hip.constant {serialization_order = 11 : i64, value = dense<2.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x2x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[COUNTS:.*]] = tensor.empty() : tensor<1x2xi32>
// CHECK-NEXT:      %[[SAI:.*]]:3 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<1x2x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x8x128xf16>)
// CHECK-SAME:          past_sequence_length(%[[PSL]] : tensor<1xi32>)
// CHECK-SAME:          outs(%[[SEL]], %[[PRESENT]] : tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          counts(%[[COUNTS]] : tensor<1x2xi32>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x2x2051xi32>, tensor<1x8x128xf16>, tensor<1x2xi32>
// CHECK-NEXT:      %[[SEL_ROWS:.*]] = tensor.collapse_shape %[[SAI]]#0 {{\[\[}}0, 1], [2]] : tensor<1x2x2051xi32> into tensor<2x2051xi32>
// CHECK-NEXT:      %[[COUNT_ROWS:.*]] = tensor.collapse_shape %[[SAI]]#2 {{\[\[}}0, 1]] : tensor<1x2xi32> into tensor<2xi32>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[DSA:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[Q]] : tensor<1x2x512xf16>) kv(%[[K]], %[[V]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[SEL_ROWS]], %[[COUNT_ROWS]], %[[SEQLENS]] : tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      return %[[DSA]]#0, %[[SAI]]#0 : tensor<1x2x512xf16>, tensor<1x2x2051xi32>
// CHECK-NEXT:    }
func.func @selection_output(
    %qk: tensor<1x2x640xf16>, %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>,
    %past: tensor<1x8x128xf16>, %psl: tensor<1xi32>,
    %q: tensor<1x2x512xf16>, %k: tensor<1x2x128xf16>, %v: tensor<1x2x128xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %seqlens: tensor<1xi32>, %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<1x2x2051xi32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %last = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %rows = "onnx.Constant"() {value = dense<[-1, 2051]> : tensor<2xi64>} : () -> tensor<2xi64>
  %qw = "onnx.Constant"() {value = dense<1.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %kw = "onnx.Constant"() {value = dense<2.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past, %none, %none, %none, %none, %psl) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x2x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, none, tensor<1x8x128xf16>, none, none, none, none, tensor<1xi32>)
    -> (tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<1x2x2051xi32>, tensor<i32>) -> tensor<1x2x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<1x2x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<1x2x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %last) {keepdims = 0 : si64} : (tensor<1x2x2051xi32>, tensor<1xi64>) -> tensor<1x2xi32>
  %sel_r = "onnx.Reshape"(%sel, %rows) {allowzero = 0 : si64} : (tensor<1x2x2051xi32>, tensor<2xi64>) -> tensor<2x2051xi32>
  %cnt_r = "onnx.Reshape"(%counts, %last) {allowzero = 0 : si64} : (tensor<1x2xi32>, tensor<1xi64>) -> tensor<2xi32>
  %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %sel_r, %cnt_r, %seqlens, %total, %none, %none, %none, %none, %none, %none) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32,
    attention_mode = "selected_only", selected_kv_source = "main"
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, none)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  return %out, %sel : tensor<1x2x512xf16>, tensor<1x2x2051xi32>
}

// The reshaped selection is also a graph output, so the ops stay unpaired.
// CHECK-LABEL: func.func @reshaped_selection_output(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x2x640xf16>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x8x128xf16>,
// CHECK-SAME:      %[[PSL:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[Q:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[K:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[V:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<2x2051xi32>) {
// CHECK-NEXT:      %[[QW:.*]] = hip.constant {serialization_order = 16 : i64, value = dense<1.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[KW:.*]] = hip.constant {serialization_order = 17 : i64, value = dense<2.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x2x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[COUNTS:.*]] = tensor.empty() : tensor<1x2xi32>
// CHECK-NEXT:      %[[SAI:.*]]:3 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<1x2x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x8x128xf16>)
// CHECK-SAME:          past_sequence_length(%[[PSL]] : tensor<1xi32>)
// CHECK-SAME:          outs(%[[SEL]], %[[PRESENT]] : tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          counts(%[[COUNTS]] : tensor<1x2xi32>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x2x2051xi32>, tensor<1x8x128xf16>, tensor<1x2xi32>
// CHECK-NEXT:      %[[SEL_ROWS:.*]] = tensor.collapse_shape %[[SAI]]#0 {{\[\[}}0, 1], [2]] : tensor<1x2x2051xi32> into tensor<2x2051xi32>
// CHECK-NEXT:      %[[COUNT_ROWS:.*]] = tensor.collapse_shape %[[SAI]]#2 {{\[\[}}0, 1]] : tensor<1x2xi32> into tensor<2xi32>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[DSA:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[Q]] : tensor<1x2x512xf16>) kv(%[[K]], %[[V]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[SEL_ROWS]], %[[COUNT_ROWS]], %[[SEQLENS]] : tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      return %[[DSA]]#0, %[[SEL_ROWS]] : tensor<1x2x512xf16>, tensor<2x2051xi32>
// CHECK-NEXT:    }
func.func @reshaped_selection_output(
    %qk: tensor<1x2x640xf16>, %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>,
    %past: tensor<1x8x128xf16>, %psl: tensor<1xi32>,
    %q: tensor<1x2x512xf16>, %k: tensor<1x2x128xf16>, %v: tensor<1x2x128xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %seqlens: tensor<1xi32>, %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<2x2051xi32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %last = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %rows = "onnx.Constant"() {value = dense<[-1, 2051]> : tensor<2xi64>} : () -> tensor<2xi64>
  %qw = "onnx.Constant"() {value = dense<1.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %kw = "onnx.Constant"() {value = dense<2.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past, %none, %none, %none, %none, %psl) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x2x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, none, tensor<1x8x128xf16>, none, none, none, none, tensor<1xi32>)
    -> (tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<1x2x2051xi32>, tensor<i32>) -> tensor<1x2x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<1x2x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<1x2x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %last) {keepdims = 0 : si64} : (tensor<1x2x2051xi32>, tensor<1xi64>) -> tensor<1x2xi32>
  %sel_r = "onnx.Reshape"(%sel, %rows) {allowzero = 0 : si64} : (tensor<1x2x2051xi32>, tensor<2xi64>) -> tensor<2x2051xi32>
  %cnt_r = "onnx.Reshape"(%counts, %last) {allowzero = 0 : si64} : (tensor<1x2xi32>, tensor<1xi64>) -> tensor<2xi32>
  %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %sel_r, %cnt_r, %seqlens, %total, %none, %none, %none, %none, %none, %none) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32,
    attention_mode = "selected_only", selected_kv_source = "main"
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, none)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  return %out, %sel_r : tensor<1x2x512xf16>, tensor<2x2051xi32>
}

// The attention reads counts from elsewhere, so the ops stay unpaired.
// CHECK-LABEL: func.func @foreign_counts(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x2x640xf16>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x8x128xf16>,
// CHECK-SAME:      %[[PSL:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[Q:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[K:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[V:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[CNT:[^:]*]]: tensor<2xi32>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> (tensor<1x2x512xf16>, tensor<1x2xi32>) {
// CHECK-NEXT:      %[[QW:.*]] = hip.constant {serialization_order = 22 : i64, value = dense<1.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[KW:.*]] = hip.constant {serialization_order = 23 : i64, value = dense<2.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x2x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[COUNTS:.*]] = tensor.empty() : tensor<1x2xi32>
// CHECK-NEXT:      %[[SAI:.*]]:3 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<1x2x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x8x128xf16>)
// CHECK-SAME:          past_sequence_length(%[[PSL]] : tensor<1xi32>)
// CHECK-SAME:          outs(%[[SEL]], %[[PRESENT]] : tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          counts(%[[COUNTS]] : tensor<1x2xi32>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x2x2051xi32>, tensor<1x8x128xf16>, tensor<1x2xi32>
// CHECK-NEXT:      %[[SEL_ROWS:.*]] = tensor.collapse_shape %[[SAI]]#0 {{\[\[}}0, 1], [2]] : tensor<1x2x2051xi32> into tensor<2x2051xi32>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[DSA:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[Q]] : tensor<1x2x512xf16>) kv(%[[K]], %[[V]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[SEL_ROWS]], %[[CNT]], %[[SEQLENS]] : tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      return %[[DSA]]#0, %[[SAI]]#2 : tensor<1x2x512xf16>, tensor<1x2xi32>
// CHECK-NEXT:    }
func.func @foreign_counts(
    %qk: tensor<1x2x640xf16>, %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>,
    %past: tensor<1x8x128xf16>, %psl: tensor<1xi32>,
    %q: tensor<1x2x512xf16>, %k: tensor<1x2x128xf16>, %v: tensor<1x2x128xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %cnt: tensor<2xi32>, %seqlens: tensor<1xi32>, %total: tensor<i32>
) -> (tensor<1x2x512xf16>, tensor<1x2xi32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %last = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %rows = "onnx.Constant"() {value = dense<[-1, 2051]> : tensor<2xi64>} : () -> tensor<2xi64>
  %qw = "onnx.Constant"() {value = dense<1.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %kw = "onnx.Constant"() {value = dense<2.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past, %none, %none, %none, %none, %psl) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x2x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, none, tensor<1x8x128xf16>, none, none, none, none, tensor<1xi32>)
    -> (tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<1x2x2051xi32>, tensor<i32>) -> tensor<1x2x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<1x2x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<1x2x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %last) {keepdims = 0 : si64} : (tensor<1x2x2051xi32>, tensor<1xi64>) -> tensor<1x2xi32>
  %sel_r = "onnx.Reshape"(%sel, %rows) {allowzero = 0 : si64} : (tensor<1x2x2051xi32>, tensor<2xi64>) -> tensor<2x2051xi32>
  %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %sel_r, %cnt, %seqlens, %total, %none, %none, %none, %none, %none, %none) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32,
    attention_mode = "selected_only", selected_kv_source = "main"
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, none)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  return %out, %counts : tensor<1x2x512xf16>, tensor<1x2xi32>
}

// A masked row may not see every position, so the ops stay unpaired.
// CHECK-LABEL: func.func @masked(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x2x640xf16>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[MASK:[^:]*]]: tensor<1x8xi64>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x8x128xf16>,
// CHECK-SAME:      %[[PSL:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[Q:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[K:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[V:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> tensor<1x2x512xf16> {
// CHECK-NEXT:      %[[QW:.*]] = hip.constant {serialization_order = 28 : i64, value = dense<1.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[KW:.*]] = hip.constant {serialization_order = 29 : i64, value = dense<2.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x2x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[COUNTS:.*]] = tensor.empty() : tensor<1x2xi32>
// CHECK-NEXT:      %[[SAI:.*]]:3 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<1x2x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x8x128xf16>)
// CHECK-SAME:          mask(%[[MASK]] : tensor<1x8xi64>)
// CHECK-SAME:          past_sequence_length(%[[PSL]] : tensor<1xi32>)
// CHECK-SAME:          outs(%[[SEL]], %[[PRESENT]] : tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          counts(%[[COUNTS]] : tensor<1x2xi32>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x2x2051xi32>, tensor<1x8x128xf16>, tensor<1x2xi32>
// CHECK-NEXT:      %[[SEL_ROWS:.*]] = tensor.collapse_shape %[[SAI]]#0 {{\[\[}}0, 1], [2]] : tensor<1x2x2051xi32> into tensor<2x2051xi32>
// CHECK-NEXT:      %[[COUNT_ROWS:.*]] = tensor.collapse_shape %[[SAI]]#2 {{\[\[}}0, 1]] : tensor<1x2xi32> into tensor<2xi32>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[DSA:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[Q]] : tensor<1x2x512xf16>) kv(%[[K]], %[[V]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[SEL_ROWS]], %[[COUNT_ROWS]], %[[SEQLENS]] : tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      return %[[DSA]]#0 : tensor<1x2x512xf16>
// CHECK-NEXT:    }
func.func @masked(
    %qk: tensor<1x2x640xf16>, %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>,
    %mask: tensor<1x8xi64>, %past: tensor<1x8x128xf16>, %psl: tensor<1xi32>,
    %q: tensor<1x2x512xf16>, %k: tensor<1x2x128xf16>, %v: tensor<1x2x128xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %seqlens: tensor<1xi32>, %total: tensor<i32>
) -> tensor<1x2x512xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %last = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %rows = "onnx.Constant"() {value = dense<[-1, 2051]> : tensor<2xi64>} : () -> tensor<2xi64>
  %qw = "onnx.Constant"() {value = dense<1.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %kw = "onnx.Constant"() {value = dense<2.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %mask, %past, %none, %none, %none, %none, %psl) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x2x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x8xi64>, tensor<1x8x128xf16>, none, none, none, none, tensor<1xi32>)
    -> (tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<1x2x2051xi32>, tensor<i32>) -> tensor<1x2x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<1x2x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<1x2x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %last) {keepdims = 0 : si64} : (tensor<1x2x2051xi32>, tensor<1xi64>) -> tensor<1x2xi32>
  %sel_r = "onnx.Reshape"(%sel, %rows) {allowzero = 0 : si64} : (tensor<1x2x2051xi32>, tensor<2xi64>) -> tensor<2x2051xi32>
  %cnt_r = "onnx.Reshape"(%counts, %last) {allowzero = 0 : si64} : (tensor<1x2xi32>, tensor<1xi64>) -> tensor<2xi32>
  %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %sel_r, %cnt_r, %seqlens, %total, %none, %none, %none, %none, %none, %none) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32,
    attention_mode = "selected_only", selected_kv_source = "main"
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, none)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  return %out : tensor<1x2x512xf16>
}

// Pairing needs past_sequence_length, so the ops stay unpaired.
// CHECK-LABEL: func.func @no_past_sequence_length(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x2x640xf16>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x6x128xf16>,
// CHECK-SAME:      %[[Q:[^:]*]]: tensor<1x2x512xf16>,
// CHECK-SAME:      %[[K:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[V:[^:]*]]: tensor<1x2x128xf16>,
// CHECK-SAME:      %[[PAST_K:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[PAST_V:[^:]*]]: tensor<1x1x16x128xf16>,
// CHECK-SAME:      %[[SEQLENS:[^:]*]]: tensor<1xi32>,
// CHECK-SAME:      %[[TOTAL:[^:]*]]: tensor<i32>) -> tensor<1x2x512xf16> {
// CHECK-NEXT:      %[[QW:.*]] = hip.constant {serialization_order = 34 : i64, value = dense<1.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[KW:.*]] = hip.constant {serialization_order = 35 : i64, value = dense<2.000000e+00> : tensor<128xf16>} : tensor<128xf16>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x2x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[COUNTS:.*]] = tensor.empty() : tensor<1x2xi32>
// CHECK-NEXT:      %[[SAI:.*]]:3 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<1x2x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x6x128xf16>)
// CHECK-SAME:          outs(%[[SEL]], %[[PRESENT]] : tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          counts(%[[COUNTS]] : tensor<1x2xi32>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x2x2051xi32>, tensor<1x8x128xf16>, tensor<1x2xi32>
// CHECK-NEXT:      %[[SEL_ROWS:.*]] = tensor.collapse_shape %[[SAI]]#0 {{\[\[}}0, 1], [2]] : tensor<1x2x2051xi32> into tensor<2x2051xi32>
// CHECK-NEXT:      %[[COUNT_ROWS:.*]] = tensor.collapse_shape %[[SAI]]#2 {{\[\[}}0, 1]] : tensor<1x2xi32> into tensor<2xi32>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.empty() : tensor<1x2x512xf16>
// CHECK-NEXT:      %[[PRESENT_K:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[PRESENT_V:.*]] = tensor.empty() : tensor<1x1x16x128xf16>
// CHECK-NEXT:      %[[DSA:.*]]:3 = hip.dynamic_sparse_attention(%[[CTX]]) ins(%[[Q]] : tensor<1x2x512xf16>) kv(%[[K]], %[[V]] : tensor<1x2x128xf16>, tensor<1x2x128xf16>) cache(%[[PAST_K]], %[[PAST_V]] : tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          select(%[[SEL_ROWS]], %[[COUNT_ROWS]], %[[SEQLENS]] : tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>)
// CHECK-SAME:          outs(%[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
// CHECK-SAME:          {kv_num_heads = 1 : i64, num_heads = 4 : i64, scale = 0.0883883461 : f32} : tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
// CHECK-NEXT:      return %[[DSA]]#0 : tensor<1x2x512xf16>
// CHECK-NEXT:    }
func.func @no_past_sequence_length(
    %qk: tensor<1x2x640xf16>, %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>,
    %past: tensor<1x6x128xf16>,
    %q: tensor<1x2x512xf16>, %k: tensor<1x2x128xf16>, %v: tensor<1x2x128xf16>,
    %past_k: tensor<1x1x16x128xf16>, %past_v: tensor<1x1x16x128xf16>,
    %seqlens: tensor<1xi32>, %total: tensor<i32>
) -> tensor<1x2x512xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %last = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %rows = "onnx.Constant"() {value = dense<[-1, 2051]> : tensor<2xi64>} : () -> tensor<2xi64>
  %qw = "onnx.Constant"() {value = dense<1.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %kw = "onnx.Constant"() {value = dense<2.0> : tensor<128xf16>} : () -> tensor<128xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x2x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, none, tensor<1x6x128xf16>)
    -> (tensor<1x2x2051xi32>, tensor<1x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<1x2x2051xi32>, tensor<i32>) -> tensor<1x2x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<1x2x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<1x2x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %last) {keepdims = 0 : si64} : (tensor<1x2x2051xi32>, tensor<1xi64>) -> tensor<1x2xi32>
  %sel_r = "onnx.Reshape"(%sel, %rows) {allowzero = 0 : si64} : (tensor<1x2x2051xi32>, tensor<2xi64>) -> tensor<2x2051xi32>
  %cnt_r = "onnx.Reshape"(%counts, %last) {allowzero = 0 : si64} : (tensor<1x2xi32>, tensor<1xi64>) -> tensor<2xi32>
  %out, %pk, %pv = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %none, %none, %sel_r, %cnt_r, %seqlens, %total, %none, %none, %none, %none, %none, %none) {
    function_name = "DynamicSparseAttention", domain_name = "com.microsoft",
    num_heads = 4 : si64, kv_num_heads = 1 : si64, scale = 0.0883883461 : f32,
    attention_mode = "selected_only", selected_kv_source = "main"
  } : (tensor<1x2x512xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>, none, none, tensor<2x2051xi32>, tensor<2xi32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, none)
    -> (tensor<1x2x512xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>)
  return %out : tensor<1x2x512xf16>
}
