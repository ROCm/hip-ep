// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x10240xf16>) -> tensor<1x10240xf16> {
    return %arg0 : tensor<1x10240xf16>
  }

  func.func @gdn_packed(
      %qkv: tensor<?x?x10240xf16>,
      %decay: tensor<?x?x48xf32>,
      %beta: tensor<?x?x48xf32>,
      %state: tensor<?x48x128x128xf32>,
      %a_log: tensor<48xf32>,
      %dt_bias: tensor<48xf32>
  ) -> (tensor<?x?x48x128xf16>, tensor<?x48x128x128xf32>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %out, %final = "onnx.Custom"(%qkv, %none, %none, %none, %decay, %beta, %state, %a_log, %dt_bias) {
      function_name = "GatedDeltaNet",
      domain_name = "com.microsoft",
      update_rule = "gated_delta",
      gate_activation = "qwen",
      beta_activation = "sigmoid",
      qk_l2_norm = 1 : si64,
      scale = 0.0 : f32,
      chunk_size = 64 : si64
    } : (tensor<?x?x10240xf16>, none, none, none, tensor<?x?x48xf32>, tensor<?x?x48xf32>, tensor<?x48x128x128xf32>, tensor<48xf32>, tensor<48xf32>)
      -> (tensor<?x?x48x128xf16>, tensor<?x48x128x128xf32>)
    return %out, %final : tensor<?x?x48x128xf16>, tensor<?x48x128x128xf32>
  }

  // CHECK-LABEL: func.func @gdn_packed
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[QKV:.*]]: tensor<?x?x10240xf16>, %[[DECAY:.*]]: tensor<?x?x48xf32>, %[[BETA:.*]]: tensor<?x?x48xf32>, %[[STATE:.*]]: tensor<?x48x128x128xf32>, %[[ALOG:.*]]: tensor<48xf32>, %[[DTB:.*]]: tensor<48xf32>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[B:.*]] = tensor.dim %[[QKV]], %[[C0]]
  // CHECK: %[[L:.*]] = tensor.dim %[[QKV]], %[[C1]]
  // CHECK: %[[OUT:.*]] = tensor.empty(%[[B]], %[[L]]) : tensor<?x?x48x128xf16>
  // CHECK: %[[SB:.*]] = tensor.dim %[[STATE]], %[[C0]]
  // CHECK: %[[FINAL:.*]] = tensor.empty(%[[SB]]) : tensor<?x48x128x128xf32>
  // CHECK: %[[R:.*]]:2 = hip.gated_delta_net(%[[CTX]]) ins(%[[QKV]], %[[DECAY]], %[[BETA]], %[[STATE]], %[[ALOG]], %[[DTB]] : tensor<?x?x10240xf16>, tensor<?x?x48xf32>, tensor<?x?x48xf32>, tensor<?x48x128x128xf32>, tensor<48xf32>, tensor<48xf32>) outs(%[[OUT]], %[[FINAL]] : tensor<?x?x48x128xf16>, tensor<?x48x128x128xf32>) : tensor<?x?x48x128xf16>, tensor<?x48x128x128xf32>
  // CHECK: return %[[R]]#0, %[[R]]#1

  func.func @gdn_gate_none(
      %qkv: tensor<1x1x10240xf16>,
      %decay: tensor<1x1x48xf32>,
      %beta: tensor<1x1x48xf32>,
      %state: tensor<1x48x128x128xf32>,
      %a_log: tensor<48xf32>,
      %dt_bias: tensor<48xf32>
  ) -> (tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %out, %final = "onnx.Custom"(%qkv, %none, %none, %none, %decay, %beta, %state, %a_log, %dt_bias) {
      function_name = "GatedDeltaNet",
      domain_name = "com.microsoft",
      update_rule = "gated_delta",
      gate_activation = "none",
      beta_activation = "sigmoid",
      qk_l2_norm = 1 : si64
    } : (tensor<1x1x10240xf16>, none, none, none, tensor<1x1x48xf32>, tensor<1x1x48xf32>, tensor<1x48x128x128xf32>, tensor<48xf32>, tensor<48xf32>)
      -> (tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>)
    return %out, %final : tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>
  }

  // CHECK-LABEL: func.func @gdn_gate_none
  // CHECK-NOT: hip.gated_delta_net
  // CHECK: "onnx.Custom"
  // CHECK-SAME: gate_activation = "none"

  func.func @gdn_unpacked(
      %q: tensor<1x1x2048xf16>,
      %k: tensor<1x1x2048xf16>,
      %v: tensor<1x1x6144xf16>,
      %decay: tensor<1x1x48xf32>,
      %beta: tensor<1x1x48xf32>,
      %state: tensor<1x48x128x128xf32>,
      %a_log: tensor<48xf32>,
      %dt_bias: tensor<48xf32>
  ) -> (tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %out, %final = "onnx.Custom"(%q, %k, %v, %none, %decay, %beta, %state, %a_log, %dt_bias) {
      function_name = "GatedDeltaNet",
      domain_name = "com.microsoft",
      update_rule = "gated_delta",
      gate_activation = "qwen",
      beta_activation = "sigmoid",
      qk_l2_norm = 1 : si64
    } : (tensor<1x1x2048xf16>, tensor<1x1x2048xf16>, tensor<1x1x6144xf16>, none, tensor<1x1x48xf32>, tensor<1x1x48xf32>, tensor<1x48x128x128xf32>, tensor<48xf32>, tensor<48xf32>)
      -> (tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>)
    return %out, %final : tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>
  }

  // CHECK-LABEL: func.func @gdn_unpacked
  // CHECK-NOT: hip.gated_delta_net
  // CHECK: "onnx.Custom"
  // CHECK-SAME: function_name = "GatedDeltaNet"

  func.func @gdn_state_capture(
      %qkv: tensor<1x1x10240xf16>,
      %decay: tensor<1x1x48xf32>,
      %beta: tensor<1x1x48xf32>,
      %state: tensor<1x48x128x128xf32>,
      %a_log: tensor<48xf32>,
      %dt_bias: tensor<48xf32>
  ) -> (tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %out, %final = "onnx.Custom"(%qkv, %none, %none, %none, %decay, %beta, %state, %a_log, %dt_bias) {
      function_name = "GatedDeltaNet",
      domain_name = "com.microsoft",
      update_rule = "gated_delta",
      gate_activation = "qwen",
      beta_activation = "sigmoid",
      qk_l2_norm = 1 : si64,
      state_update_capacity = 2 : si64
    } : (tensor<1x1x10240xf16>, none, none, none, tensor<1x1x48xf32>, tensor<1x1x48xf32>, tensor<1x48x128x128xf32>, tensor<48xf32>, tensor<48xf32>)
      -> (tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>)
    return %out, %final : tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>
  }

  // CHECK-LABEL: func.func @gdn_state_capture
  // CHECK-NOT: hip.gated_delta_net
  // CHECK: "onnx.Custom"
  // CHECK-SAME: state_update_capacity = 2 : si64

  func.func @gdn_no_l2_norm(
      %qkv: tensor<1x1x10240xf16>,
      %decay: tensor<1x1x48xf32>,
      %beta: tensor<1x1x48xf32>,
      %state: tensor<1x48x128x128xf32>,
      %a_log: tensor<48xf32>,
      %dt_bias: tensor<48xf32>
  ) -> (tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %out, %final = "onnx.Custom"(%qkv, %none, %none, %none, %decay, %beta, %state, %a_log, %dt_bias) {
      function_name = "GatedDeltaNet",
      domain_name = "com.microsoft",
      update_rule = "gated_delta",
      gate_activation = "qwen",
      beta_activation = "sigmoid",
      qk_l2_norm = 0 : si64
    } : (tensor<1x1x10240xf16>, none, none, none, tensor<1x1x48xf32>, tensor<1x1x48xf32>, tensor<1x48x128x128xf32>, tensor<48xf32>, tensor<48xf32>)
      -> (tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>)
    return %out, %final : tensor<1x1x48x128xf16>, tensor<1x48x128x128xf32>
  }

  // CHECK-LABEL: func.func @gdn_no_l2_norm
  // CHECK-NOT: hip.gated_delta_net
  // CHECK: "onnx.Custom"
  // CHECK-SAME: qk_l2_norm = 0 : si64
}
