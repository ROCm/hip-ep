// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // A shared cache can be larger than the logical prefix. Pin the capacity
  // rule with past dim 16 and total_seq_len 8.
  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: %[[PK:[^,]+]]: tensor<1x2x16x4xf16>
  // CHECK-SAME: %[[PV:[^,]+]]: tensor<1x2x16x4xf16>
  // CHECK: %[[TOTAL_I32:.*]] = arith.constant 8 : i32
  // CHECK: %[[TOTAL:.*]] = arith.index_cast %[[TOTAL_I32]] : i32 to index
  // CHECK: %[[PK_DIM:.*]] = tensor.dim %[[PK]]
  // CHECK: %[[PK_CAP:.*]] = arith.maxui %[[PK_DIM]], %[[TOTAL]] : index
  // CHECK: %[[PV_DIM:.*]] = tensor.dim %[[PV]]
  // CHECK: %[[PV_CAP:.*]] = arith.maxui %[[PV_DIM]], %[[TOTAL]] : index
  // CHECK: tensor.empty(%[[PK_CAP]]) : tensor<1x2x?x4xf16>
  // CHECK: tensor.empty(%[[PV_CAP]]) : tensor<1x2x?x4xf16>
  func.func @main_graph(
      %query: tensor<1x1x16xf16>,
      %key: tensor<1x1x8xf16>,
      %value: tensor<1x1x8xf16>,
      %past_key: tensor<1x2x16x4xf16>,
      %past_value: tensor<1x2x16x4xf16>,
      %seqlens_k: tensor<1xi32>)
      -> (tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
          tensor<1x2x?x4xf16>) {
    %total = arith.constant dense<8> : tensor<i32>
    %out:3 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                           %seqlens_k, %total)
        {domain_name = "com.microsoft", function_name = "GroupQueryAttention",
         kv_num_heads = 2 : si64, num_heads = 4 : si64}
        : (tensor<1x1x16xf16>, tensor<1x1x8xf16>, tensor<1x1x8xf16>,
           tensor<1x2x16x4xf16>, tensor<1x2x16x4xf16>, tensor<1xi32>,
           tensor<i32>)
        -> (tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
            tensor<1x2x?x4xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
          tensor<1x2x?x4xf16>
  }

  // A non-shared growing cache can have a smaller past buffer. Pin the same
  // max rule with past dim 4 and total_seq_len 8.
  // CHECK-LABEL: func.func @growing_non_shared_total_exceeds_past
  // CHECK-SAME: %[[GROW_PK:[^,]+]]: tensor<1x2x4x4xf16>
  // CHECK-SAME: %[[GROW_PV:[^,]+]]: tensor<1x2x4x4xf16>
  // CHECK: %[[GROW_TOTAL_I32:.*]] = arith.constant 8 : i32
  // CHECK: %[[GROW_TOTAL:.*]] = arith.index_cast %[[GROW_TOTAL_I32]] : i32 to index
  // CHECK: %[[GROW_PK_DIM:.*]] = tensor.dim %[[GROW_PK]]
  // CHECK: %[[GROW_PK_CAP:.*]] = arith.maxui %[[GROW_PK_DIM]], %[[GROW_TOTAL]] : index
  // CHECK: %[[GROW_PV_DIM:.*]] = tensor.dim %[[GROW_PV]]
  // CHECK: %[[GROW_PV_CAP:.*]] = arith.maxui %[[GROW_PV_DIM]], %[[GROW_TOTAL]] : index
  // CHECK: tensor.empty(%[[GROW_PK_CAP]]) : tensor<1x2x?x4xf16>
  // CHECK: tensor.empty(%[[GROW_PV_CAP]]) : tensor<1x2x?x4xf16>
  func.func @growing_non_shared_total_exceeds_past(
      %query: tensor<1x1x16xf16>,
      %key: tensor<1x1x8xf16>,
      %value: tensor<1x1x8xf16>,
      %past_key: tensor<1x2x4x4xf16>,
      %past_value: tensor<1x2x4x4xf16>,
      %seqlens_k: tensor<1xi32>)
      -> (tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
          tensor<1x2x?x4xf16>) {
    %total = arith.constant dense<8> : tensor<i32>
    %out:3 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                           %seqlens_k, %total)
        {domain_name = "com.microsoft", function_name = "GroupQueryAttention",
         kv_num_heads = 2 : si64, num_heads = 4 : si64}
        : (tensor<1x1x16xf16>, tensor<1x1x8xf16>, tensor<1x1x8xf16>,
           tensor<1x2x4x4xf16>, tensor<1x2x4x4xf16>, tensor<1xi32>,
           tensor<i32>)
        -> (tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
            tensor<1x2x?x4xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
          tensor<1x2x?x4xf16>
  }

  // Without past, both dynamic present outputs use logical total_seq_len
  // directly and share its one synchronized readback.
  // CHECK-LABEL: func.func @no_past_uses_logical_total
  // CHECK-SAME: %[[NOPAST_CTX:.*]]: !hip.context
  // CHECK-SAME: %[[NOPAST_TOTAL:[^,)]+]]: tensor<i32>
  // CHECK: %[[NOPAST_I32:.*]] = hip.readback_scalar(%[[NOPAST_CTX]], %[[NOPAST_TOTAL]] : tensor<i32>) -> i32
  // CHECK: %[[NOPAST_IDX:.*]] = arith.index_cast %[[NOPAST_I32]] : i32 to index
  // CHECK-NOT: hip.readback_scalar
  // CHECK-NOT: arith.maxui
  // CHECK: tensor.empty(%[[NOPAST_IDX]]) : tensor<1x2x?x4xf16>
  // CHECK: tensor.empty(%[[NOPAST_IDX]]) : tensor<1x2x?x4xf16>
  // CHECK: hip.gqa
  func.func @no_past_uses_logical_total(
      %query: tensor<1x1x16xf16>,
      %key: tensor<1x1x8xf16>,
      %value: tensor<1x1x8xf16>,
      %seqlens_k: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
          tensor<1x2x?x4xf16>) {
    %none_key = "onnx.NoValue"() {value} : () -> none
    %none_value = "onnx.NoValue"() {value} : () -> none
    %out:3 = "onnx.Custom"(%query, %key, %value, %none_key, %none_value,
                           %seqlens_k, %total)
        {domain_name = "com.microsoft", function_name = "GroupQueryAttention",
         kv_num_heads = 2 : si64, num_heads = 4 : si64}
        : (tensor<1x1x16xf16>, tensor<1x1x8xf16>, tensor<1x1x8xf16>,
           none, none, tensor<1xi32>, tensor<i32>)
        -> (tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
            tensor<1x2x?x4xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
          tensor<1x2x?x4xf16>
  }

  // Matching key/value capacities remain independent while both max
  // operations reuse exactly one logical total_seq_len readback.
  // CHECK-LABEL: func.func @two_present_outputs_reuse_logical_readback
  // CHECK-SAME: %[[REUSE_CTX:.*]]: !hip.context
  // CHECK-SAME: %[[REUSE_PK:[^,]+]]: tensor<1x2x?x4xf16>
  // CHECK-SAME: %[[REUSE_PV:[^,]+]]: tensor<1x2x?x4xf16>
  // CHECK-SAME: %[[REUSE_TOTAL:[^,)]+]]: tensor<i32>
  // CHECK: %[[REUSE_I32:.*]] = hip.readback_scalar(%[[REUSE_CTX]], %[[REUSE_TOTAL]] : tensor<i32>) -> i32
  // CHECK: %[[REUSE_IDX:.*]] = arith.index_cast %[[REUSE_I32]] : i32 to index
  // CHECK-NOT: hip.readback_scalar
  // CHECK: %[[REUSE_PK_DIM:.*]] = tensor.dim %[[REUSE_PK]]
  // CHECK: %[[REUSE_PK_CAP:.*]] = arith.maxui %[[REUSE_PK_DIM]], %[[REUSE_IDX]] : index
  // CHECK: %[[REUSE_PV_DIM:.*]] = tensor.dim %[[REUSE_PV]]
  // CHECK: %[[REUSE_PV_CAP:.*]] = arith.maxui %[[REUSE_PV_DIM]], %[[REUSE_IDX]] : index
  // CHECK: tensor.empty(%[[REUSE_PK_CAP]]) : tensor<1x2x?x4xf16>
  // CHECK: tensor.empty(%[[REUSE_PV_CAP]]) : tensor<1x2x?x4xf16>
  func.func @two_present_outputs_reuse_logical_readback(
      %query: tensor<1x1x16xf16>,
      %key: tensor<1x1x8xf16>,
      %value: tensor<1x1x8xf16>,
      %past_key: tensor<1x2x?x4xf16>,
      %past_value: tensor<1x2x?x4xf16>,
      %seqlens_k: tensor<1xi32>,
      %total: tensor<i32>)
      -> (tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
          tensor<1x2x?x4xf16>) {
    %out:3 = "onnx.Custom"(%query, %key, %value, %past_key, %past_value,
                           %seqlens_k, %total)
        {domain_name = "com.microsoft", function_name = "GroupQueryAttention",
         kv_num_heads = 2 : si64, num_heads = 4 : si64}
        : (tensor<1x1x16xf16>, tensor<1x1x8xf16>, tensor<1x1x8xf16>,
           tensor<1x2x?x4xf16>, tensor<1x2x?x4xf16>, tensor<1xi32>,
           tensor<i32>)
        -> (tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
            tensor<1x2x?x4xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x16xf16>, tensor<1x2x?x4xf16>,
          tensor<1x2x?x4xf16>
  }

}
