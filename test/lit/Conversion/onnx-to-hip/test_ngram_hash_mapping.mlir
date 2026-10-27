// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x16xi64>) -> tensor<1x16xi64> {
    return %arg0 : tensor<1x16xi64>
  }

  func.func @ngram_engram(
      %ids: tensor<?x?xi64>,
      %mult: tensor<3xi64>,
      %vocab: tensor<16xi64>,
      %past: tensor<?x2xi64>,
      %off: tensor<16xi64>,
      %eos: tensor<i64>
  ) -> (tensor<?x?x16xi64>, tensor<?x2xi64>) {
    %hash, %present = "onnx.Custom"(%ids, %mult, %vocab, %past, %off, %eos) {
      function_name = "NGramHashMapping",
      domain_name = "com.microsoft",
      max_ngram_size = 3 : si64,
      n_head_per_ngram = 8 : si64,
      pad_id = 248044 : si64,
      reset_on_eos = 1 : si64
    } : (tensor<?x?xi64>, tensor<3xi64>, tensor<16xi64>, tensor<?x2xi64>, tensor<16xi64>, tensor<i64>)
      -> (tensor<?x?x16xi64>, tensor<?x2xi64>)
    return %hash, %present : tensor<?x?x16xi64>, tensor<?x2xi64>
  }

  // CHECK-LABEL: func.func @ngram_engram
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IDS:.*]]: tensor<?x?xi64>, %[[MULT:.*]]: tensor<3xi64>, %[[VOCAB:.*]]: tensor<16xi64>, %[[PAST:.*]]: tensor<?x2xi64>, %[[OFF:.*]]: tensor<16xi64>, %[[EOS:.*]]: tensor<i64>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[B:.*]] = tensor.dim %[[IDS]], %[[C0]]
  // CHECK: %[[S:.*]] = tensor.dim %[[IDS]], %[[C1]]
  // CHECK: %[[HASH:.*]] = tensor.empty(%[[B]], %[[S]]) : tensor<?x?x16xi64>
  // CHECK: %[[B2:.*]] = tensor.dim %[[IDS]], %[[C0]]
  // CHECK: %[[PRESENT:.*]] = tensor.empty(%[[B2]]) : tensor<?x2xi64>
  // CHECK: %[[R:.*]]:2 = hip.ngram_hash_mapping(%[[CTX]]) ins(%[[IDS]], %[[MULT]], %[[VOCAB]] : tensor<?x?xi64>, tensor<3xi64>, tensor<16xi64>) past_ids(%[[PAST]] : tensor<?x2xi64>) head_offsets(%[[OFF]] : tensor<16xi64>) eos_token_id(%[[EOS]] : tensor<i64>) outs(%[[HASH]], %[[PRESENT]] : tensor<?x?x16xi64>, tensor<?x2xi64>) {max_ngram_size = 3 : i64, n_head_per_ngram = 8 : i64, pad_id = 248044 : i64, reset_on_eos = 1 : i64}
  // CHECK: return %[[R]]#0, %[[R]]#1

  func.func @ngram_minimal(
      %ids: tensor<2x4xi32>,
      %mult: tensor<3xi32>,
      %vocab: tensor<4xi32>
  ) -> tensor<2x4x4xi32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %hash = "onnx.Custom"(%ids, %mult, %vocab, %none, %none, %none, %none) {
      function_name = "NGramHashMapping",
      domain_name = "com.microsoft",
      max_ngram_size = 3 : si64,
      n_head_per_ngram = 2 : si64,
      pad_id = 9 : si64
    } : (tensor<2x4xi32>, tensor<3xi32>, tensor<4xi32>, none, none, none, none)
      -> tensor<2x4x4xi32>
    return %hash : tensor<2x4x4xi32>
  }

  // CHECK-LABEL: func.func @ngram_minimal
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IDS:.*]]: tensor<2x4xi32>, %[[MULT:.*]]: tensor<3xi32>, %[[VOCAB:.*]]: tensor<4xi32>)
  // CHECK: %[[HASH:.*]] = tensor.empty() : tensor<2x4x4xi32>
  // CHECK: hip.ngram_hash_mapping(%[[CTX]]) ins(%[[IDS]], %[[MULT]], %[[VOCAB]] : tensor<2x4xi32>, tensor<3xi32>, tensor<4xi32>) outs(%[[HASH]] : tensor<2x4x4xi32>) {max_ngram_size = 3 : i64, n_head_per_ngram = 2 : i64, pad_id = 9 : i64} : tensor<2x4x4xi32>

  func.func @ngram_segment_ids(
      %ids: tensor<1x4xi64>,
      %mult: tensor<2xi64>,
      %vocab: tensor<2xi64>,
      %seg: tensor<1x4xi32>
  ) -> tensor<1x4x2xi64> {
    %none = "onnx.NoValue"() {value} : () -> none
    %hash, %present = "onnx.Custom"(%ids, %mult, %vocab, %none, %none, %none, %seg) {
      function_name = "NGramHashMapping",
      domain_name = "com.microsoft",
      max_ngram_size = 2 : si64,
      n_head_per_ngram = 2 : si64,
      pad_id = 0 : si64
    } : (tensor<1x4xi64>, tensor<2xi64>, tensor<2xi64>, none, none, none, tensor<1x4xi32>)
      -> (tensor<1x4x2xi64>, none)
    return %hash : tensor<1x4x2xi64>
  }

  // CHECK-LABEL: func.func @ngram_segment_ids
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IDS:.*]]: tensor<1x4xi64>, %[[MULT:.*]]: tensor<2xi64>, %[[VOCAB:.*]]: tensor<2xi64>, %[[SEG:.*]]: tensor<1x4xi32>)
  // CHECK: %[[HASH:.*]] = tensor.empty() : tensor<1x4x2xi64>
  // CHECK: hip.ngram_hash_mapping(%[[CTX]]) ins(%[[IDS]], %[[MULT]], %[[VOCAB]] : tensor<1x4xi64>, tensor<2xi64>, tensor<2xi64>) segment_ids(%[[SEG]] : tensor<1x4xi32>) outs(%[[HASH]] : tensor<1x4x2xi64>) {max_ngram_size = 2 : i64, n_head_per_ngram = 2 : i64, pad_id = 0 : i64} : tensor<1x4x2xi64>

  func.func @ngram_i32_pad_out_of_range(
      %ids: tensor<2x4xi32>,
      %mult: tensor<3xi32>,
      %vocab: tensor<4xi32>
  ) -> tensor<2x4x4xi32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %hash = "onnx.Custom"(%ids, %mult, %vocab, %none, %none, %none, %none) {
      function_name = "NGramHashMapping",
      domain_name = "com.microsoft",
      max_ngram_size = 3 : si64,
      n_head_per_ngram = 2 : si64,
      pad_id = 4294967296 : si64
    } : (tensor<2x4xi32>, tensor<3xi32>, tensor<4xi32>, none, none, none, none)
      -> tensor<2x4x4xi32>
    return %hash : tensor<2x4x4xi32>
  }

  // CHECK-LABEL: func.func @ngram_i32_pad_out_of_range
  // CHECK: "onnx.Custom"
  // CHECK-NOT: hip.ngram_hash_mapping
}
