// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

// The conversion requires a @main_graph entry function.
func.func @main_graph(%arg0: tensor<1xf16>) -> tensor<1xf16> {
  return %arg0 : tensor<1xf16>
}

// ReduceSum(Where(GreaterOrEqual(sel, 0), 1, 0)) becomes the indexer's counts
// result. The returned selection still reads result 0.
// CHECK-LABEL: func.func @fold_counts(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<2x3x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<2x8x128xf16>,
// CHECK-SAME:      %[[PSL:[^:]*]]: tensor<1xi32>) -> (tensor<2x3x2051xi32>, tensor<2x8x128xf16>, tensor<2x3xi32>) {
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<2x8x128xf16>
// CHECK-NEXT:      %[[COUNTS:.*]] = tensor.empty() : tensor<2x3xi32>
// CHECK-NEXT:      %[[R:.*]]:3 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<2x3x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<2x8x128xf16>)
// CHECK-SAME:          past_sequence_length(%[[PSL]] : tensor<1xi32>) outs(%[[SEL]], %[[PRESENT]] : tensor<2x3x2051xi32>, tensor<2x8x128xf16>) counts(%[[COUNTS]] : tensor<2x3xi32>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<2x3x2051xi32>, tensor<2x8x128xf16>, tensor<2x3xi32>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#1, %[[R]]#2 : tensor<2x3x2051xi32>, tensor<2x8x128xf16>, tensor<2x3xi32>
// CHECK-NEXT:    }
func.func @fold_counts(
    %qk: tensor<2x3x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>,
    %past: tensor<2x8x128xf16>, %psl: tensor<1xi32>
) -> (tensor<2x3x2051xi32>, tensor<2x8x128xf16>, tensor<2x3xi32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %axes = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past, %none, %none, %none, %none, %psl) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<2x3x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, none, tensor<2x8x128xf16>, none, none, none, none, tensor<1xi32>)
    -> (tensor<2x3x2051xi32>, tensor<2x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<2x3x2051xi32>, tensor<i32>) -> tensor<2x3x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<2x3x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %axes) {keepdims = 0 : si64} : (tensor<2x3x2051xi32>, tensor<1xi64>) -> tensor<2x3xi32>
  return %sel, %present, %counts : tensor<2x3x2051xi32>, tensor<2x8x128xf16>, tensor<2x3xi32>
}

// Reductions over axis -1 and axis 2 share one counts result.
// The rotary table fold applies to the same indexer.
// CHECK-LABEL: func.func @fold_counts_shared_with_rotary(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<2x3x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[IDS:[^:]*]]: tensor<3x2x8xi64>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<2x5x128xf16>) -> (tensor<2x3x2051xi32>, tensor<2x3xi32>, tensor<2x3xi32>) {
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<2x8x128xf16>
// CHECK-NEXT:      %[[COUNTS:.*]] = tensor.empty() : tensor<2x3xi32>
// CHECK-NEXT:      %[[R:.*]]:3 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<2x3x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<16x4xf16>, tensor<16x4xf16>, tensor<2x5x128xf16>)
// CHECK-SAME:          rotary_position_ids(%[[IDS]] : tensor<3x2x8xi64>) outs(%[[SEL]], %[[PRESENT]] : tensor<2x3x2051xi32>, tensor<2x8x128xf16>) counts(%[[COUNTS]] : tensor<2x3xi32>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, rotary_sections = array<i32: 0, 1, 0, 0>, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<2x3x2051xi32>, tensor<2x8x128xf16>, tensor<2x3xi32>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#2, %[[R]]#2 : tensor<2x3x2051xi32>, tensor<2x3xi32>, tensor<2x3xi32>
// CHECK-NEXT:    }
func.func @fold_counts_shared_with_rotary(
    %qk: tensor<2x3x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %ids: tensor<3x2x8xi64>, %cos_base: tensor<16x4xf16>,
    %sin_base: tensor<16x4xf16>, %past: tensor<2x5x128xf16>
) -> (tensor<2x3x2051xi32>, tensor<2x3xi32>, tensor<2x3xi32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %k0 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %k1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %m = "onnx.Constant"() {value = dense<[false, true, false, false]> : tensor<4xi1>} : () -> tensor<4xi1>
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %last = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %two = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %p0 = "onnx.Gather"(%ids, %k0) {axis = 0 : si64} : (tensor<3x2x8xi64>, tensor<i64>) -> tensor<2x8xi64>
  %p1 = "onnx.Gather"(%ids, %k1) {axis = 0 : si64} : (tensor<3x2x8xi64>, tensor<i64>) -> tensor<2x8xi64>
  %cg0 = "onnx.Gather"(%cos_base, %p0) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<2x8xi64>) -> tensor<2x8x4xf16>
  %cg1 = "onnx.Gather"(%cos_base, %p1) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<2x8xi64>) -> tensor<2x8x4xf16>
  %cw = "onnx.Where"(%m, %cg1, %cg0) : (tensor<4xi1>, tensor<2x8x4xf16>, tensor<2x8x4xf16>) -> tensor<2x8x4xf16>
  %cos = "onnx.Concat"(%cw, %cw) {axis = -1 : si64} : (tensor<2x8x4xf16>, tensor<2x8x4xf16>) -> tensor<2x8x8xf16>
  %sg0 = "onnx.Gather"(%sin_base, %p0) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<2x8xi64>) -> tensor<2x8x4xf16>
  %sg1 = "onnx.Gather"(%sin_base, %p1) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<2x8xi64>) -> tensor<2x8x4xf16>
  %sw = "onnx.Where"(%m, %sg1, %sg0) : (tensor<4xi1>, tensor<2x8x4xf16>, tensor<2x8x4xf16>) -> tensor<2x8x4xf16>
  %sin = "onnx.Concat"(%sw, %sw) {axis = -1 : si64} : (tensor<2x8x4xf16>, tensor<2x8x4xf16>) -> tensor<2x8x8xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<2x3x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<2x8x8xf16>, tensor<2x8x8xf16>, none, tensor<2x5x128xf16>)
    -> (tensor<2x3x2051xi32>, tensor<2x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<2x3x2051xi32>, tensor<i32>) -> tensor<2x3x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<2x3x2051xi32>
  %counts_last = "onnx.ReduceSum"(%w, %last) {keepdims = 0 : si64} : (tensor<2x3x2051xi32>, tensor<1xi64>) -> tensor<2x3xi32>
  %counts_two = "onnx.ReduceSum"(%w, %two) {keepdims = 0 : si64} : (tensor<2x3x2051xi32>, tensor<1xi64>) -> tensor<2x3xi32>
  return %sel, %counts_last, %counts_two : tensor<2x3x2051xi32>, tensor<2x3xi32>, tensor<2x3xi32>
}

// Threshold 1 does not count token 0, so the glue stays.
// CHECK-LABEL: func.func @no_fold_threshold(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<2x3x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<2x5x128xf16>) -> tensor<2x3xi32> {
// CHECK-NEXT:      %[[ZERO:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<0> : tensor<i32>} : tensor<i32>
// CHECK-NEXT:      %[[ONE:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<1> : tensor<i32>} : tensor<i32>
// CHECK-NEXT:      %[[AXES:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<-1> : tensor<1xi64>} : tensor<1xi64>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<2x8x128xf16>
// CHECK-NEXT:      %[[R:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<2x3x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<2x5x128xf16>) outs(%[[SEL]], %[[PRESENT]] : tensor<2x3x2051xi32>, tensor<2x8x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<2x3x2051xi32>, tensor<2x8x128xf16>
// CHECK-NEXT:      %[[LT_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[LT:.*]] = hip.less(%[[CTX]]) ins(%[[R]]#0, %[[ONE]] : tensor<2x3x2051xi32>, tensor<i32>) outs(%[[LT_INIT]] : tensor<2x3x2051xi1>) : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[GE_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[GE:.*]] = hip.not(%[[CTX]]) ins(%[[LT]] : tensor<2x3x2051xi1>) outs(%[[GE_INIT]] : tensor<2x3x2051xi1>) : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[W_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[W:.*]] = hip.where(%[[CTX]]) ins(%[[GE]], %[[ONE]], %[[ZERO]] : tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) outs(%[[W_INIT]] : tensor<2x3x2051xi32>) : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[COUNTS_INIT:.*]] = tensor.empty() : tensor<2x3xi32>
// CHECK-NEXT:      %[[COUNTS:.*]] = hip.reduce_sum(%[[CTX]]) ins(%[[W]], %[[AXES]] : tensor<2x3x2051xi32>, tensor<1xi64>) outs(%[[COUNTS_INIT]] : tensor<2x3xi32>) {keepdims = 0 : i64, reduced_axes = array<i64: 2>} : tensor<2x3xi32>
// CHECK-NEXT:      return %[[COUNTS]] : tensor<2x3xi32>
// CHECK-NEXT:    }
func.func @no_fold_threshold(
    %qk: tensor<2x3x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>, %past: tensor<2x5x128xf16>
) -> tensor<2x3xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %axes = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<2x3x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, none, tensor<2x5x128xf16>)
    -> (tensor<2x3x2051xi32>, tensor<2x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %one) : (tensor<2x3x2051xi32>, tensor<i32>) -> tensor<2x3x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<2x3x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %axes) {keepdims = 0 : si64} : (tensor<2x3x2051xi32>, tensor<1xi64>) -> tensor<2x3xi32>
  return %counts : tensor<2x3xi32>
}

// Where(..., 2, 0) sums to twice the count.
// CHECK-LABEL: func.func @no_fold_where_values(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<2x3x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<2x5x128xf16>) -> tensor<2x3xi32> {
// CHECK-NEXT:      %[[ZERO:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<0> : tensor<i32>} : tensor<i32>
// CHECK-NEXT:      %[[TWO:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<2> : tensor<i32>} : tensor<i32>
// CHECK-NEXT:      %[[AXES:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<-1> : tensor<1xi64>} : tensor<1xi64>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<2x8x128xf16>
// CHECK-NEXT:      %[[R:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<2x3x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<2x5x128xf16>) outs(%[[SEL]], %[[PRESENT]] : tensor<2x3x2051xi32>, tensor<2x8x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<2x3x2051xi32>, tensor<2x8x128xf16>
// CHECK-NEXT:      %[[LT_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[LT:.*]] = hip.less(%[[CTX]]) ins(%[[R]]#0, %[[ZERO]] : tensor<2x3x2051xi32>, tensor<i32>) outs(%[[LT_INIT]] : tensor<2x3x2051xi1>) : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[GE_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[GE:.*]] = hip.not(%[[CTX]]) ins(%[[LT]] : tensor<2x3x2051xi1>) outs(%[[GE_INIT]] : tensor<2x3x2051xi1>) : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[W_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[W:.*]] = hip.where(%[[CTX]]) ins(%[[GE]], %[[TWO]], %[[ZERO]] : tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) outs(%[[W_INIT]] : tensor<2x3x2051xi32>) : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[COUNTS_INIT:.*]] = tensor.empty() : tensor<2x3xi32>
// CHECK-NEXT:      %[[COUNTS:.*]] = hip.reduce_sum(%[[CTX]]) ins(%[[W]], %[[AXES]] : tensor<2x3x2051xi32>, tensor<1xi64>) outs(%[[COUNTS_INIT]] : tensor<2x3xi32>) {keepdims = 0 : i64, reduced_axes = array<i64: 2>} : tensor<2x3xi32>
// CHECK-NEXT:      return %[[COUNTS]] : tensor<2x3xi32>
// CHECK-NEXT:    }
func.func @no_fold_where_values(
    %qk: tensor<2x3x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>, %past: tensor<2x5x128xf16>
) -> tensor<2x3xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %two = "onnx.Constant"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %axes = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<2x3x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, none, tensor<2x5x128xf16>)
    -> (tensor<2x3x2051xi32>, tensor<2x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<2x3x2051xi32>, tensor<i32>) -> tensor<2x3x2051xi1>
  %w = "onnx.Where"(%ge, %two, %zero) : (tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<2x3x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %axes) {keepdims = 0 : si64} : (tensor<2x3x2051xi32>, tensor<1xi64>) -> tensor<2x3xi32>
  return %counts : tensor<2x3xi32>
}

// keepdims = 1 gives [B, S, 1], which the indexer does not produce.
// CHECK-LABEL: func.func @no_fold_keepdims(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<2x3x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<2x5x128xf16>) -> tensor<2x3x1xi32> {
// CHECK-NEXT:      %[[ZERO:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<0> : tensor<i32>} : tensor<i32>
// CHECK-NEXT:      %[[ONE:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<1> : tensor<i32>} : tensor<i32>
// CHECK-NEXT:      %[[AXES:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<-1> : tensor<1xi64>} : tensor<1xi64>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<2x8x128xf16>
// CHECK-NEXT:      %[[R:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<2x3x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<2x5x128xf16>) outs(%[[SEL]], %[[PRESENT]] : tensor<2x3x2051xi32>, tensor<2x8x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<2x3x2051xi32>, tensor<2x8x128xf16>
// CHECK-NEXT:      %[[LT_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[LT:.*]] = hip.less(%[[CTX]]) ins(%[[R]]#0, %[[ZERO]] : tensor<2x3x2051xi32>, tensor<i32>) outs(%[[LT_INIT]] : tensor<2x3x2051xi1>) : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[GE_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[GE:.*]] = hip.not(%[[CTX]]) ins(%[[LT]] : tensor<2x3x2051xi1>) outs(%[[GE_INIT]] : tensor<2x3x2051xi1>) : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[W_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[W:.*]] = hip.where(%[[CTX]]) ins(%[[GE]], %[[ONE]], %[[ZERO]] : tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) outs(%[[W_INIT]] : tensor<2x3x2051xi32>) : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[COUNTS_INIT:.*]] = tensor.empty() : tensor<2x3x1xi32>
// CHECK-NEXT:      %[[COUNTS:.*]] = hip.reduce_sum(%[[CTX]]) ins(%[[W]], %[[AXES]] : tensor<2x3x2051xi32>, tensor<1xi64>) outs(%[[COUNTS_INIT]] : tensor<2x3x1xi32>) {reduced_axes = array<i64: 2>} : tensor<2x3x1xi32>
// CHECK-NEXT:      return %[[COUNTS]] : tensor<2x3x1xi32>
// CHECK-NEXT:    }
func.func @no_fold_keepdims(
    %qk: tensor<2x3x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>, %past: tensor<2x5x128xf16>
) -> tensor<2x3x1xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %axes = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<2x3x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, none, tensor<2x5x128xf16>)
    -> (tensor<2x3x2051xi32>, tensor<2x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<2x3x2051xi32>, tensor<i32>) -> tensor<2x3x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<2x3x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %axes) {keepdims = 1 : si64} : (tensor<2x3x2051xi32>, tensor<1xi64>) -> tensor<2x3x1xi32>
  return %counts : tensor<2x3x1xi32>
}

// Reducing axis 1 sums across query rows, not within one row.
// CHECK-LABEL: func.func @no_fold_other_axis(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<2x3x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<8x64xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<2x5x128xf16>) -> tensor<2x2051xi32> {
// CHECK-NEXT:      %[[ZERO:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<0> : tensor<i32>} : tensor<i32>
// CHECK-NEXT:      %[[ONE:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<1> : tensor<i32>} : tensor<i32>
// CHECK-NEXT:      %[[AXES:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<1> : tensor<1xi64>} : tensor<1xi64>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<2x8x128xf16>
// CHECK-NEXT:      %[[R:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<2x3x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, tensor<2x5x128xf16>) outs(%[[SEL]], %[[PRESENT]] : tensor<2x3x2051xi32>, tensor<2x8x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<2x3x2051xi32>, tensor<2x8x128xf16>
// CHECK-NEXT:      %[[LT_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[LT:.*]] = hip.less(%[[CTX]]) ins(%[[R]]#0, %[[ZERO]] : tensor<2x3x2051xi32>, tensor<i32>) outs(%[[LT_INIT]] : tensor<2x3x2051xi1>) : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[GE_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[GE:.*]] = hip.not(%[[CTX]]) ins(%[[LT]] : tensor<2x3x2051xi1>) outs(%[[GE_INIT]] : tensor<2x3x2051xi1>) : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[W_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[W:.*]] = hip.where(%[[CTX]]) ins(%[[GE]], %[[ONE]], %[[ZERO]] : tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) outs(%[[W_INIT]] : tensor<2x3x2051xi32>) : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[COUNTS_INIT:.*]] = tensor.empty() : tensor<2x2051xi32>
// CHECK-NEXT:      %[[COUNTS:.*]] = hip.reduce_sum(%[[CTX]]) ins(%[[W]], %[[AXES]] : tensor<2x3x2051xi32>, tensor<1xi64>) outs(%[[COUNTS_INIT]] : tensor<2x2051xi32>) {keepdims = 0 : i64, reduced_axes = array<i64: 1>} : tensor<2x2051xi32>
// CHECK-NEXT:      return %[[COUNTS]] : tensor<2x2051xi32>
// CHECK-NEXT:    }
func.func @no_fold_other_axis(
    %qk: tensor<2x3x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>, %past: tensor<2x5x128xf16>
) -> tensor<2x2051xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<2x3x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<8x64xf16>, tensor<8x64xf16>, none, tensor<2x5x128xf16>)
    -> (tensor<2x3x2051xi32>, tensor<2x8x128xf16>)
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<2x3x2051xi32>, tensor<i32>) -> tensor<2x3x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<2x3x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %axes) {keepdims = 0 : si64} : (tensor<2x3x2051xi32>, tensor<1xi64>) -> tensor<2x2051xi32>
  return %counts : tensor<2x2051xi32>
}

// The counted tensor does not come from an indexer.
// CHECK-LABEL: func.func @no_fold_not_indexer(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[SEL:[^:]*]]: tensor<2x3x2051xi32>) -> tensor<2x3xi32> {
// CHECK-NEXT:      %[[ZERO:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<0> : tensor<i32>} : tensor<i32>
// CHECK-NEXT:      %[[ONE:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<1> : tensor<i32>} : tensor<i32>
// CHECK-NEXT:      %[[AXES:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<-1> : tensor<1xi64>} : tensor<1xi64>
// CHECK-NEXT:      %[[LT_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[LT:.*]] = hip.less(%[[CTX]]) ins(%[[SEL]], %[[ZERO]] : tensor<2x3x2051xi32>, tensor<i32>) outs(%[[LT_INIT]] : tensor<2x3x2051xi1>) : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[GE_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[GE:.*]] = hip.not(%[[CTX]]) ins(%[[LT]] : tensor<2x3x2051xi1>) outs(%[[GE_INIT]] : tensor<2x3x2051xi1>) : tensor<2x3x2051xi1>
// CHECK-NEXT:      %[[W_INIT:.*]] = tensor.empty() : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[W:.*]] = hip.where(%[[CTX]]) ins(%[[GE]], %[[ONE]], %[[ZERO]] : tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) outs(%[[W_INIT]] : tensor<2x3x2051xi32>) : tensor<2x3x2051xi32>
// CHECK-NEXT:      %[[COUNTS_INIT:.*]] = tensor.empty() : tensor<2x3xi32>
// CHECK-NEXT:      %[[COUNTS:.*]] = hip.reduce_sum(%[[CTX]]) ins(%[[W]], %[[AXES]] : tensor<2x3x2051xi32>, tensor<1xi64>) outs(%[[COUNTS_INIT]] : tensor<2x3xi32>) {keepdims = 0 : i64, reduced_axes = array<i64: 2>} : tensor<2x3xi32>
// CHECK-NEXT:      return %[[COUNTS]] : tensor<2x3xi32>
// CHECK-NEXT:    }
func.func @no_fold_not_indexer(%sel: tensor<2x3x2051xi32>) -> tensor<2x3xi32> {
  %zero = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %one = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %axes = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %ge = "onnx.GreaterOrEqual"(%sel, %zero) : (tensor<2x3x2051xi32>, tensor<i32>) -> tensor<2x3x2051xi1>
  %w = "onnx.Where"(%ge, %one, %zero) : (tensor<2x3x2051xi1>, tensor<i32>, tensor<i32>) -> tensor<2x3x2051xi32>
  %counts = "onnx.ReduceSum"(%w, %axes) {keepdims = 0 : si64} : (tensor<2x3x2051xi32>, tensor<1xi64>) -> tensor<2x3xi32>
  return %counts : tensor<2x3xi32>
}
