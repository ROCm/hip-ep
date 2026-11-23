// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

// The conversion requires a @main_graph entry function.
func.func @main_graph(%arg0: tensor<1xf16>) -> tensor<1xf16> {
  return %arg0 : tensor<1xf16>
}

// Two indexers share a three-section table subgraph. Both read the base tables
// through the ids with one section map, and the Gather/Where/Concat ops are
// gone.
// CHECK-LABEL: func.func @fold_shared_tables(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK0:[^:]*]]: tensor<1x1x640xf16>,
// CHECK-SAME:      %[[QK1:[^:]*]]: tensor<1x1x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[IDS:[^:]*]]: tensor<3x1x16xi64>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[PAST0:[^:]*]]: tensor<1x16x128xf16>,
// CHECK-SAME:      %[[PAST1:[^:]*]]: tensor<1x16x128xf16>,
// CHECK-SAME:      %[[PSL:[^:]*]]: tensor<1xi32>) -> (tensor<1x1x2051xi32>, tensor<1x16x128xf16>, tensor<1x1x2051xi32>, tensor<1x16x128xf16>) {
// CHECK-NEXT:      %[[SEL0:.*]] = tensor.empty() : tensor<1x1x2051xi32>
// CHECK-NEXT:      %[[PRESENT0:.*]] = tensor.empty() : tensor<1x16x128xf16>
// CHECK-NEXT:      %[[R0:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK0]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST0]] : tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<16x4xf16>, tensor<16x4xf16>, tensor<1x16x128xf16>)
// CHECK-SAME:          past_sequence_length(%[[PSL]] : tensor<1xi32>) rotary_position_ids(%[[IDS]] : tensor<3x1x16xi64>) outs(%[[SEL0]], %[[PRESENT0]] : tensor<1x1x2051xi32>, tensor<1x16x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, rotary_sections = array<i32: 0, 1, 2, 0>, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x1x2051xi32>, tensor<1x16x128xf16>
// CHECK-NEXT:      %[[SEL1:.*]] = tensor.empty() : tensor<1x1x2051xi32>
// CHECK-NEXT:      %[[PRESENT1:.*]] = tensor.empty() : tensor<1x16x128xf16>
// CHECK-NEXT:      %[[R1:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK1]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST1]] : tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<16x4xf16>, tensor<16x4xf16>, tensor<1x16x128xf16>)
// CHECK-SAME:          past_sequence_length(%[[PSL]] : tensor<1xi32>) rotary_position_ids(%[[IDS]] : tensor<3x1x16xi64>) outs(%[[SEL1]], %[[PRESENT1]] : tensor<1x1x2051xi32>, tensor<1x16x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, rotary_sections = array<i32: 0, 1, 2, 0>, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x1x2051xi32>, tensor<1x16x128xf16>
// CHECK-NEXT:      return %[[R0]]#0, %[[R0]]#1, %[[R1]]#0, %[[R1]]#1 : tensor<1x1x2051xi32>, tensor<1x16x128xf16>, tensor<1x1x2051xi32>, tensor<1x16x128xf16>
// CHECK-NEXT:    }
func.func @fold_shared_tables(
    %qk0: tensor<1x1x640xf16>, %qk1: tensor<1x1x640xf16>,
    %qw: tensor<128xf16>, %kw: tensor<128xf16>, %ids: tensor<3x1x16xi64>,
    %cos_base: tensor<16x4xf16>, %sin_base: tensor<16x4xf16>,
    %past0: tensor<1x16x128xf16>, %past1: tensor<1x16x128xf16>,
    %psl: tensor<1xi32>
) -> (tensor<1x1x2051xi32>, tensor<1x16x128xf16>, tensor<1x1x2051xi32>, tensor<1x16x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %k0 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %k1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %k2 = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
  %m1 = "onnx.Constant"() {value = dense<[false, true, false, false]> : tensor<4xi1>} : () -> tensor<4xi1>
  %m2 = "onnx.Constant"() {value = dense<[false, false, true, false]> : tensor<4xi1>} : () -> tensor<4xi1>
  %p0 = "onnx.Gather"(%ids, %k0) {axis = 0 : si64} : (tensor<3x1x16xi64>, tensor<i64>) -> tensor<1x16xi64>
  %p1 = "onnx.Gather"(%ids, %k1) {axis = 0 : si64} : (tensor<3x1x16xi64>, tensor<i64>) -> tensor<1x16xi64>
  %p2 = "onnx.Gather"(%ids, %k2) {axis = 0 : si64} : (tensor<3x1x16xi64>, tensor<i64>) -> tensor<1x16xi64>
  %cg0 = "onnx.Gather"(%cos_base, %p0) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x16xi64>) -> tensor<1x16x4xf16>
  %cg1 = "onnx.Gather"(%cos_base, %p1) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x16xi64>) -> tensor<1x16x4xf16>
  %cg2 = "onnx.Gather"(%cos_base, %p2) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x16xi64>) -> tensor<1x16x4xf16>
  %cw1 = "onnx.Where"(%m1, %cg1, %cg0) : (tensor<4xi1>, tensor<1x16x4xf16>, tensor<1x16x4xf16>) -> tensor<1x16x4xf16>
  %cw2 = "onnx.Where"(%m2, %cg2, %cw1) : (tensor<4xi1>, tensor<1x16x4xf16>, tensor<1x16x4xf16>) -> tensor<1x16x4xf16>
  %cos = "onnx.Concat"(%cw2, %cw2) {axis = -1 : si64} : (tensor<1x16x4xf16>, tensor<1x16x4xf16>) -> tensor<1x16x8xf16>
  %sg0 = "onnx.Gather"(%sin_base, %p0) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x16xi64>) -> tensor<1x16x4xf16>
  %sg1 = "onnx.Gather"(%sin_base, %p1) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x16xi64>) -> tensor<1x16x4xf16>
  %sg2 = "onnx.Gather"(%sin_base, %p2) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x16xi64>) -> tensor<1x16x4xf16>
  %sw1 = "onnx.Where"(%m1, %sg1, %sg0) : (tensor<4xi1>, tensor<1x16x4xf16>, tensor<1x16x4xf16>) -> tensor<1x16x4xf16>
  %sw2 = "onnx.Where"(%m2, %sg2, %sw1) : (tensor<4xi1>, tensor<1x16x4xf16>, tensor<1x16x4xf16>) -> tensor<1x16x4xf16>
  %sin = "onnx.Concat"(%sw2, %sw2) {axis = -1 : si64} : (tensor<1x16x4xf16>, tensor<1x16x4xf16>) -> tensor<1x16x8xf16>
  %sel0, %present0 = "onnx.Custom"(%qk0, %none, %qw, %kw, %cos, %sin, %none, %past0, %none, %none, %none, %none, %psl) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x1x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<1x16x8xf16>, tensor<1x16x8xf16>, none, tensor<1x16x128xf16>, none, none, none, none, tensor<1xi32>)
    -> (tensor<1x1x2051xi32>, tensor<1x16x128xf16>)
  %sel1, %present1 = "onnx.Custom"(%qk1, %none, %qw, %kw, %cos, %sin, %none, %past1, %none, %none, %none, %none, %psl) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x1x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<1x16x8xf16>, tensor<1x16x8xf16>, none, tensor<1x16x128xf16>, none, none, none, none, tensor<1xi32>)
    -> (tensor<1x1x2051xi32>, tensor<1x16x128xf16>)
  return %sel0, %present0, %sel1, %present1 : tensor<1x1x2051xi32>, tensor<1x16x128xf16>, tensor<1x1x2051xi32>, tensor<1x16x128xf16>
}

// Gather(ids, -1) reads the last section, so the map uses section 2.
// This indexer has no past_sequence_length, and Concat uses axis 2.
// CHECK-LABEL: func.func @fold_negative_section_index(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x1x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[IDS:[^:]*]]: tensor<3x1x8xi64>,
// CHECK-SAME:      %[[COS:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[SIN:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x7x128xf16>) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x1x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[R:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<16x4xf16>, tensor<16x4xf16>, tensor<1x7x128xf16>)
// CHECK-SAME:          rotary_position_ids(%[[IDS]] : tensor<3x1x8xi64>) outs(%[[SEL]], %[[PRESENT]] : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, rotary_sections = array<i32: 0, 2, 2, 0>, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:    }
func.func @fold_negative_section_index(
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %ids: tensor<3x1x8xi64>, %cos_base: tensor<16x4xf16>,
    %sin_base: tensor<16x4xf16>, %past: tensor<1x7x128xf16>
) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %k0 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %kl = "onnx.Constant"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
  %m = "onnx.Constant"() {value = dense<[false, true, true, false]> : tensor<4xi1>} : () -> tensor<4xi1>
  %p0 = "onnx.Gather"(%ids, %k0) {axis = 0 : si64} : (tensor<3x1x8xi64>, tensor<i64>) -> tensor<1x8xi64>
  %pl = "onnx.Gather"(%ids, %kl) {axis = 0 : si64} : (tensor<3x1x8xi64>, tensor<i64>) -> tensor<1x8xi64>
  %cg0 = "onnx.Gather"(%cos_base, %p0) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %cgl = "onnx.Gather"(%cos_base, %pl) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %cw = "onnx.Where"(%m, %cgl, %cg0) : (tensor<4xi1>, tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x4xf16>
  %cos = "onnx.Concat"(%cw, %cw) {axis = 2 : si64} : (tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x8xf16>
  %sg0 = "onnx.Gather"(%sin_base, %p0) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %sgl = "onnx.Gather"(%sin_base, %pl) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %sw = "onnx.Where"(%m, %sgl, %sg0) : (tensor<4xi1>, tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x4xf16>
  %sin = "onnx.Concat"(%sw, %sw) {axis = 2 : si64} : (tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x8xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x1x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<1x8x8xf16>, tensor<1x8x8xf16>, none, tensor<1x7x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
  return %sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// The sin table picks its sections with another mask than cos, so the tables
// cannot share one section map. Both stay materialized.
// CHECK-LABEL: func.func @no_fold_mismatched_sections(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x1x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[IDS:[^:]*]]: tensor<3x1x8xi64>,
// CHECK-SAME:      %[[COS_BASE:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[SIN_BASE:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x7x128xf16>) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
// CHECK-NEXT:      %[[K0:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<0> : tensor<i64>} : tensor<i64>
// CHECK-NEXT:      %[[K1:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<1> : tensor<i64>} : tensor<i64>
// CHECK-NEXT:      %[[COS_MASK:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<[false, true, false, false]> : tensor<4xi1>} : tensor<4xi1>
// CHECK-NEXT:      %[[SIN_MASK:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<[false, false, false, true]> : tensor<4xi1>} : tensor<4xi1>
// CHECK-NEXT:      %[[P0_INIT:.*]] = tensor.empty() : tensor<1x8xi64>
// CHECK-NEXT:      %[[P0:.*]] = hip.gather(%[[CTX]]) ins(%[[IDS]], %[[K0]] : tensor<3x1x8xi64>, tensor<i64>) outs(%[[P0_INIT]] : tensor<1x8xi64>) : tensor<1x8xi64>
// CHECK-NEXT:      %[[P1_INIT:.*]] = tensor.empty() : tensor<1x8xi64>
// CHECK-NEXT:      %[[P1:.*]] = hip.gather(%[[CTX]]) ins(%[[IDS]], %[[K1]] : tensor<3x1x8xi64>, tensor<i64>) outs(%[[P1_INIT]] : tensor<1x8xi64>) : tensor<1x8xi64>
// CHECK-NEXT:      %[[CG0_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[CG0:.*]] = hip.gather(%[[CTX]]) ins(%[[COS_BASE]], %[[P0]] : tensor<16x4xf16>, tensor<1x8xi64>) outs(%[[CG0_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[CG1_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[CG1:.*]] = hip.gather(%[[CTX]]) ins(%[[COS_BASE]], %[[P1]] : tensor<16x4xf16>, tensor<1x8xi64>) outs(%[[CG1_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[CW_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[CW:.*]] = hip.where(%[[CTX]]) ins(%[[COS_MASK]], %[[CG1]], %[[CG0]] : tensor<4xi1>, tensor<1x8x4xf16>, tensor<1x8x4xf16>) outs(%[[CW_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[COS_INIT:.*]] = tensor.empty() : tensor<1x8x8xf16>
// CHECK-NEXT:      %[[COS_LO:.*]] = tensor.insert_slice %[[CW]] into %[[COS_INIT]][0, 0, 0] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[COS:.*]] = tensor.insert_slice %[[CW]] into %[[COS_LO]][0, 0, 4] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[SG0_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[SG0:.*]] = hip.gather(%[[CTX]]) ins(%[[SIN_BASE]], %[[P0]] : tensor<16x4xf16>, tensor<1x8xi64>) outs(%[[SG0_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[SG1_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[SG1:.*]] = hip.gather(%[[CTX]]) ins(%[[SIN_BASE]], %[[P1]] : tensor<16x4xf16>, tensor<1x8xi64>) outs(%[[SG1_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[SW_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[SW:.*]] = hip.where(%[[CTX]]) ins(%[[SIN_MASK]], %[[SG1]], %[[SG0]] : tensor<4xi1>, tensor<1x8x4xf16>, tensor<1x8x4xf16>) outs(%[[SW_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[SIN_INIT:.*]] = tensor.empty() : tensor<1x8x8xf16>
// CHECK-NEXT:      %[[SIN_LO:.*]] = tensor.insert_slice %[[SW]] into %[[SIN_INIT]][0, 0, 0] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[SIN:.*]] = tensor.insert_slice %[[SW]] into %[[SIN_LO]][0, 0, 4] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x1x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[R:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<1x8x8xf16>, tensor<1x8x8xf16>, tensor<1x7x128xf16>) outs(%[[SEL]], %[[PRESENT]] : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:    }
func.func @no_fold_mismatched_sections(
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %ids: tensor<3x1x8xi64>, %cos_base: tensor<16x4xf16>,
    %sin_base: tensor<16x4xf16>, %past: tensor<1x7x128xf16>
) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %k0 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %k1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %mc = "onnx.Constant"() {value = dense<[false, true, false, false]> : tensor<4xi1>} : () -> tensor<4xi1>
  %ms = "onnx.Constant"() {value = dense<[false, false, false, true]> : tensor<4xi1>} : () -> tensor<4xi1>
  %p0 = "onnx.Gather"(%ids, %k0) {axis = 0 : si64} : (tensor<3x1x8xi64>, tensor<i64>) -> tensor<1x8xi64>
  %p1 = "onnx.Gather"(%ids, %k1) {axis = 0 : si64} : (tensor<3x1x8xi64>, tensor<i64>) -> tensor<1x8xi64>
  %cg0 = "onnx.Gather"(%cos_base, %p0) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %cg1 = "onnx.Gather"(%cos_base, %p1) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %cw = "onnx.Where"(%mc, %cg1, %cg0) : (tensor<4xi1>, tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x4xf16>
  %cos = "onnx.Concat"(%cw, %cw) {axis = -1 : si64} : (tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x8xf16>
  %sg0 = "onnx.Gather"(%sin_base, %p0) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %sg1 = "onnx.Gather"(%sin_base, %p1) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %sw = "onnx.Where"(%ms, %sg1, %sg0) : (tensor<4xi1>, tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x4xf16>
  %sin = "onnx.Concat"(%sw, %sw) {axis = -1 : si64} : (tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x8xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x1x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<1x8x8xf16>, tensor<1x8x8xf16>, none, tensor<1x7x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
  return %sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// A run-time mask leaves the section of each frequency unknown.
// CHECK-LABEL: func.func @no_fold_runtime_mask(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x1x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[IDS:[^:]*]]: tensor<3x1x8xi64>,
// CHECK-SAME:      %[[BASE:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[MASK:[^:]*]]: tensor<4xi1>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x7x128xf16>) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
// CHECK-NEXT:      %[[K0:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<0> : tensor<i64>} : tensor<i64>
// CHECK-NEXT:      %[[K1:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<1> : tensor<i64>} : tensor<i64>
// CHECK-NEXT:      %[[P0_INIT:.*]] = tensor.empty() : tensor<1x8xi64>
// CHECK-NEXT:      %[[P0:.*]] = hip.gather(%[[CTX]]) ins(%[[IDS]], %[[K0]] : tensor<3x1x8xi64>, tensor<i64>) outs(%[[P0_INIT]] : tensor<1x8xi64>) : tensor<1x8xi64>
// CHECK-NEXT:      %[[P1_INIT:.*]] = tensor.empty() : tensor<1x8xi64>
// CHECK-NEXT:      %[[P1:.*]] = hip.gather(%[[CTX]]) ins(%[[IDS]], %[[K1]] : tensor<3x1x8xi64>, tensor<i64>) outs(%[[P1_INIT]] : tensor<1x8xi64>) : tensor<1x8xi64>
// CHECK-NEXT:      %[[G0_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[G0:.*]] = hip.gather(%[[CTX]]) ins(%[[BASE]], %[[P0]] : tensor<16x4xf16>, tensor<1x8xi64>) outs(%[[G0_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[G1_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[G1:.*]] = hip.gather(%[[CTX]]) ins(%[[BASE]], %[[P1]] : tensor<16x4xf16>, tensor<1x8xi64>) outs(%[[G1_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[W_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[W:.*]] = hip.where(%[[CTX]]) ins(%[[MASK]], %[[G1]], %[[G0]] : tensor<4xi1>, tensor<1x8x4xf16>, tensor<1x8x4xf16>) outs(%[[W_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[TABLE_INIT:.*]] = tensor.empty() : tensor<1x8x8xf16>
// CHECK-NEXT:      %[[TABLE_LO:.*]] = tensor.insert_slice %[[W]] into %[[TABLE_INIT]][0, 0, 0] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[TABLE:.*]] = tensor.insert_slice %[[W]] into %[[TABLE_LO]][0, 0, 4] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x1x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[R:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[TABLE]], %[[TABLE]], %[[PAST]] : tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<1x8x8xf16>, tensor<1x8x8xf16>, tensor<1x7x128xf16>) outs(%[[SEL]], %[[PRESENT]] : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:    }
func.func @no_fold_runtime_mask(
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %ids: tensor<3x1x8xi64>, %base: tensor<16x4xf16>, %mask: tensor<4xi1>,
    %past: tensor<1x7x128xf16>
) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %k0 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %k1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %p0 = "onnx.Gather"(%ids, %k0) {axis = 0 : si64} : (tensor<3x1x8xi64>, tensor<i64>) -> tensor<1x8xi64>
  %p1 = "onnx.Gather"(%ids, %k1) {axis = 0 : si64} : (tensor<3x1x8xi64>, tensor<i64>) -> tensor<1x8xi64>
  %g0 = "onnx.Gather"(%base, %p0) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %g1 = "onnx.Gather"(%base, %p1) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %w = "onnx.Where"(%mask, %g1, %g0) : (tensor<4xi1>, tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x4xf16>
  %table = "onnx.Concat"(%w, %w) {axis = -1 : si64} : (tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x8xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %table, %table, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x1x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<1x8x8xf16>, tensor<1x8x8xf16>, none, tensor<1x7x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
  return %sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// Section 3 is past the 3 sections of the ids.
// CHECK-LABEL: func.func @no_fold_section_out_of_range(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x1x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[IDS:[^:]*]]: tensor<3x1x8xi64>,
// CHECK-SAME:      %[[BASE:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x7x128xf16>) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
// CHECK-NEXT:      %[[K0:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<0> : tensor<i64>} : tensor<i64>
// CHECK-NEXT:      %[[K3:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<3> : tensor<i64>} : tensor<i64>
// CHECK-NEXT:      %[[MASK:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<[false, true, false, false]> : tensor<4xi1>} : tensor<4xi1>
// CHECK-NEXT:      %[[P0_INIT:.*]] = tensor.empty() : tensor<1x8xi64>
// CHECK-NEXT:      %[[P0:.*]] = hip.gather(%[[CTX]]) ins(%[[IDS]], %[[K0]] : tensor<3x1x8xi64>, tensor<i64>) outs(%[[P0_INIT]] : tensor<1x8xi64>) : tensor<1x8xi64>
// CHECK-NEXT:      %[[P3_INIT:.*]] = tensor.empty() : tensor<1x8xi64>
// CHECK-NEXT:      %[[P3:.*]] = hip.gather(%[[CTX]]) ins(%[[IDS]], %[[K3]] : tensor<3x1x8xi64>, tensor<i64>) outs(%[[P3_INIT]] : tensor<1x8xi64>) : tensor<1x8xi64>
// CHECK-NEXT:      %[[G0_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[G0:.*]] = hip.gather(%[[CTX]]) ins(%[[BASE]], %[[P0]] : tensor<16x4xf16>, tensor<1x8xi64>) outs(%[[G0_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[G3_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[G3:.*]] = hip.gather(%[[CTX]]) ins(%[[BASE]], %[[P3]] : tensor<16x4xf16>, tensor<1x8xi64>) outs(%[[G3_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[W_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[W:.*]] = hip.where(%[[CTX]]) ins(%[[MASK]], %[[G3]], %[[G0]] : tensor<4xi1>, tensor<1x8x4xf16>, tensor<1x8x4xf16>) outs(%[[W_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[TABLE_INIT:.*]] = tensor.empty() : tensor<1x8x8xf16>
// CHECK-NEXT:      %[[TABLE_LO:.*]] = tensor.insert_slice %[[W]] into %[[TABLE_INIT]][0, 0, 0] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[TABLE:.*]] = tensor.insert_slice %[[W]] into %[[TABLE_LO]][0, 0, 4] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x1x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[R:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[TABLE]], %[[TABLE]], %[[PAST]] : tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<1x8x8xf16>, tensor<1x8x8xf16>, tensor<1x7x128xf16>) outs(%[[SEL]], %[[PRESENT]] : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:    }
func.func @no_fold_section_out_of_range(
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %ids: tensor<3x1x8xi64>, %base: tensor<16x4xf16>, %past: tensor<1x7x128xf16>
) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %k0 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %k3 = "onnx.Constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
  %m = "onnx.Constant"() {value = dense<[false, true, false, false]> : tensor<4xi1>} : () -> tensor<4xi1>
  %p0 = "onnx.Gather"(%ids, %k0) {axis = 0 : si64} : (tensor<3x1x8xi64>, tensor<i64>) -> tensor<1x8xi64>
  %p3 = "onnx.Gather"(%ids, %k3) {axis = 0 : si64} : (tensor<3x1x8xi64>, tensor<i64>) -> tensor<1x8xi64>
  %g0 = "onnx.Gather"(%base, %p0) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %g3 = "onnx.Gather"(%base, %p3) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %w = "onnx.Where"(%m, %g3, %g0) : (tensor<4xi1>, tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x4xf16>
  %table = "onnx.Concat"(%w, %w) {axis = -1 : si64} : (tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x8xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %table, %table, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x1x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<1x8x8xf16>, tensor<1x8x8xf16>, none, tensor<1x7x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
  return %sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// The cos and sin tables gather their positions from different ids.
// CHECK-LABEL: func.func @no_fold_distinct_ids(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x1x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[COS_IDS:[^:]*]]: tensor<1x1x8xi64>,
// CHECK-SAME:      %[[SIN_IDS:[^:]*]]: tensor<1x1x8xi64>,
// CHECK-SAME:      %[[BASE:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x7x128xf16>) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
// CHECK-NEXT:      %[[K0:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<0> : tensor<i64>} : tensor<i64>
// CHECK-NEXT:      %[[CP_INIT:.*]] = tensor.empty() : tensor<1x8xi64>
// CHECK-NEXT:      %[[CP:.*]] = hip.gather(%[[CTX]]) ins(%[[COS_IDS]], %[[K0]] : tensor<1x1x8xi64>, tensor<i64>) outs(%[[CP_INIT]] : tensor<1x8xi64>) : tensor<1x8xi64>
// CHECK-NEXT:      %[[CG_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[CG:.*]] = hip.gather(%[[CTX]]) ins(%[[BASE]], %[[CP]] : tensor<16x4xf16>, tensor<1x8xi64>) outs(%[[CG_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[COS_INIT:.*]] = tensor.empty() : tensor<1x8x8xf16>
// CHECK-NEXT:      %[[COS_LO:.*]] = tensor.insert_slice %[[CG]] into %[[COS_INIT]][0, 0, 0] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[COS:.*]] = tensor.insert_slice %[[CG]] into %[[COS_LO]][0, 0, 4] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[SP_INIT:.*]] = tensor.empty() : tensor<1x8xi64>
// CHECK-NEXT:      %[[SP:.*]] = hip.gather(%[[CTX]]) ins(%[[SIN_IDS]], %[[K0]] : tensor<1x1x8xi64>, tensor<i64>) outs(%[[SP_INIT]] : tensor<1x8xi64>) : tensor<1x8xi64>
// CHECK-NEXT:      %[[SG_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[SG:.*]] = hip.gather(%[[CTX]]) ins(%[[BASE]], %[[SP]] : tensor<16x4xf16>, tensor<1x8xi64>) outs(%[[SG_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[SIN_INIT:.*]] = tensor.empty() : tensor<1x8x8xf16>
// CHECK-NEXT:      %[[SIN_LO:.*]] = tensor.insert_slice %[[SG]] into %[[SIN_INIT]][0, 0, 0] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[SIN:.*]] = tensor.insert_slice %[[SG]] into %[[SIN_LO]][0, 0, 4] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x1x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[R:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[PAST]] : tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<1x8x8xf16>, tensor<1x8x8xf16>, tensor<1x7x128xf16>) outs(%[[SEL]], %[[PRESENT]] : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:    }
func.func @no_fold_distinct_ids(
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos_ids: tensor<1x1x8xi64>, %sin_ids: tensor<1x1x8xi64>,
    %base: tensor<16x4xf16>, %past: tensor<1x7x128xf16>
) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %k0 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %cp = "onnx.Gather"(%cos_ids, %k0) {axis = 0 : si64} : (tensor<1x1x8xi64>, tensor<i64>) -> tensor<1x8xi64>
  %cg = "onnx.Gather"(%base, %cp) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %cos = "onnx.Concat"(%cg, %cg) {axis = -1 : si64} : (tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x8xf16>
  %sp = "onnx.Gather"(%sin_ids, %k0) {axis = 0 : si64} : (tensor<1x1x8xi64>, tensor<i64>) -> tensor<1x8xi64>
  %sg = "onnx.Gather"(%base, %sp) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %sin = "onnx.Concat"(%sg, %sg) {axis = -1 : si64} : (tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x8xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %cos, %sin, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x1x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<1x8x8xf16>, tensor<1x8x8xf16>, none, tensor<1x7x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
  return %sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// Concat of two different halves is not a duplicated half table.
// CHECK-LABEL: func.func @no_fold_distinct_halves(
// CHECK-SAME:      %[[CTX:[^:]*]]: !hip.context,
// CHECK-SAME:      %[[QK:[^:]*]]: tensor<1x1x640xf16>,
// CHECK-SAME:      %[[QW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[KW:[^:]*]]: tensor<128xf16>,
// CHECK-SAME:      %[[IDS:[^:]*]]: tensor<1x1x8xi64>,
// CHECK-SAME:      %[[BASE:[^:]*]]: tensor<16x4xf16>,
// CHECK-SAME:      %[[HALF:[^:]*]]: tensor<1x8x4xf16>,
// CHECK-SAME:      %[[PAST:[^:]*]]: tensor<1x7x128xf16>) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
// CHECK-NEXT:      %[[K0:.*]] = hip.constant {serialization_order = {{[0-9]+}} : i64, value = dense<0> : tensor<i64>} : tensor<i64>
// CHECK-NEXT:      %[[P_INIT:.*]] = tensor.empty() : tensor<1x8xi64>
// CHECK-NEXT:      %[[P:.*]] = hip.gather(%[[CTX]]) ins(%[[IDS]], %[[K0]] : tensor<1x1x8xi64>, tensor<i64>) outs(%[[P_INIT]] : tensor<1x8xi64>) : tensor<1x8xi64>
// CHECK-NEXT:      %[[G_INIT:.*]] = tensor.empty() : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[G:.*]] = hip.gather(%[[CTX]]) ins(%[[BASE]], %[[P]] : tensor<16x4xf16>, tensor<1x8xi64>) outs(%[[G_INIT]] : tensor<1x8x4xf16>) : tensor<1x8x4xf16>
// CHECK-NEXT:      %[[TABLE_INIT:.*]] = tensor.empty() : tensor<1x8x8xf16>
// CHECK-NEXT:      %[[TABLE_LO:.*]] = tensor.insert_slice %[[G]] into %[[TABLE_INIT]][0, 0, 0] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[TABLE:.*]] = tensor.insert_slice %[[HALF]] into %[[TABLE_LO]][0, 0, 4] [1, 8, 4] [1, 1, 1] : tensor<1x8x4xf16> into tensor<1x8x8xf16>
// CHECK-NEXT:      %[[SEL:.*]] = tensor.empty() : tensor<1x1x2051xi32>
// CHECK-NEXT:      %[[PRESENT:.*]] = tensor.empty() : tensor<1x8x128xf16>
// CHECK-NEXT:      %[[R:.*]]:2 = hip.sparse_attention_indexer(%[[CTX]]) ins(%[[QK]], %[[QW]], %[[KW]], %[[TABLE]], %[[TABLE]], %[[PAST]] : tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>, tensor<1x8x8xf16>, tensor<1x8x8xf16>, tensor<1x7x128xf16>) outs(%[[SEL]], %[[PRESENT]] : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
// CHECK-SAME:          {compress_ratio = 4 : i64, epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32, token_budget = 2048 : i64} : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:      return %[[R]]#0, %[[R]]#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
// CHECK-NEXT:    }
func.func @no_fold_distinct_halves(
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %ids: tensor<1x1x8xi64>, %base: tensor<16x4xf16>, %half: tensor<1x8x4xf16>,
    %past: tensor<1x7x128xf16>
) -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %k0 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %p = "onnx.Gather"(%ids, %k0) {axis = 0 : si64} : (tensor<1x1x8xi64>, tensor<i64>) -> tensor<1x8xi64>
  %g = "onnx.Gather"(%base, %p) {axis = 0 : si64} : (tensor<16x4xf16>, tensor<1x8xi64>) -> tensor<1x8x4xf16>
  %table = "onnx.Concat"(%g, %half) {axis = -1 : si64} : (tensor<1x8x4xf16>, tensor<1x8x4xf16>) -> tensor<1x8x8xf16>
  %sel, %present = "onnx.Custom"(%qk, %none, %qw, %kw, %table, %table, %none, %past) {
    function_name = "SparseAttentionIndexer", domain_name = "com.microsoft",
    policy_mode = "qsa", compress_ratio = 4 : si64, token_budget = 2048 : si64
  } : (tensor<1x1x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<1x8x8xf16>, tensor<1x8x8xf16>, none, tensor<1x7x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
  return %sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}
