// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --test-hip-whole-shape-dim-reify %s | FileCheck %s

// CHECK-LABEL: func.func @dynamic_batch_and_sequences
// CHECK-SAME: %[[QUERY:[^,]+]]: tensor<?x?x128xf16>
// CHECK-DAG: %[[B:.*]] = tensor.dim %[[QUERY]], %{{.*}}
// CHECK-DAG: %[[SQ:.*]] = tensor.dim %[[QUERY]], %{{.*}}
// CHECK: return %[[B]], %[[SQ]], %{{.*}} : index, index, index
func.func @dynamic_batch_and_sequences(
    %ctx: !hip.context,
    %query: tensor<?x?x128xf16>,
    %key: tensor<?x?x128xf16>,
    %value: tensor<?x?x128xf16>,
    %output: tensor<?x?x128xf16>) -> (index, index, index) {
  %result = hip.multi_head_attention(%ctx)
      ins(%query, %key, %value :
          tensor<?x?x128xf16>, tensor<?x?x128xf16>,
          tensor<?x?x128xf16>)
      outs(%output : tensor<?x?x128xf16>)
      {num_heads = 8 : i64}
      : tensor<?x?x128xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %b = tensor.dim %result, %c0 : tensor<?x?x128xf16>
  %sq = tensor.dim %result, %c1 : tensor<?x?x128xf16>
  %hidden = tensor.dim %result, %c2 : tensor<?x?x128xf16>
  return %b, %sq, %hidden : index, index, index
}

// CHECK-LABEL: func.func @value_hidden_source
// CHECK-SAME: %[[V:[^,]+]]: tensor<1x16x?xf16>
// CHECK: %[[H:.*]] = tensor.dim %[[V]], %{{.*}}
// CHECK: return %[[H]] : index
func.func @value_hidden_source(
    %ctx: !hip.context, %q: tensor<1x8x128xf16>,
    %k: tensor<1x16x128xf16>, %v: tensor<1x16x?xf16>,
    %out: tensor<1x8x?xf16>) -> index {
  %r = hip.multi_head_attention(%ctx)
      ins(%q, %k, %v : tensor<1x8x128xf16>, tensor<1x16x128xf16>, tensor<1x16x?xf16>)
      outs(%out : tensor<1x8x?xf16>) {num_heads = 8 : i64} : tensor<1x8x?xf16>
  %c2 = arith.constant 2 : index
  %h = tensor.dim %r, %c2 : tensor<1x8x?xf16>
  return %h : index
}

// CHECK-LABEL: func.func @packed_kv_query_hidden
// CHECK-SAME: %[[Q:[^,]+]]: tensor<1x8x?xf16>
// CHECK: %[[H:.*]] = tensor.dim %[[Q]], %{{.*}}
// CHECK: return %[[H]] : index
func.func @packed_kv_query_hidden(
    %ctx: !hip.context, %q: tensor<1x8x?xf16>,
    %kv: tensor<1x16x8x2x?xf16>, %out: tensor<1x8x?xf16>) -> index {
  %r = hip.multi_head_attention(%ctx)
      ins(%q, %kv : tensor<1x8x?xf16>, tensor<1x16x8x2x?xf16>)
      outs(%out : tensor<1x8x?xf16>) {num_heads = 8 : i64} : tensor<1x8x?xf16>
  %c2 = arith.constant 2 : index
  %h = tensor.dim %r, %c2 : tensor<1x8x?xf16>
  return %h : index
}

// CHECK-LABEL: func.func @packed_qkv_hidden_product
// CHECK-SAME: %[[Q:[^,]+]]: tensor<1x8x8x3x?xf16>
// CHECK: %[[D:.*]] = tensor.dim %[[Q]], %{{.*}}
// CHECK: %[[H:.*]] = arith.muli %[[D]], %{{.*}} : index
// CHECK: return %[[H]] : index
func.func @packed_qkv_hidden_product(
    %ctx: !hip.context, %qkv: tensor<1x8x8x3x?xf16>,
    %out: tensor<1x8x?xf16>) -> index {
  %r = hip.multi_head_attention(%ctx)
      ins(%qkv : tensor<1x8x8x3x?xf16>)
      outs(%out : tensor<1x8x?xf16>) {num_heads = 8 : i64} : tensor<1x8x?xf16>
  %c2 = arith.constant 2 : index
  %h = tensor.dim %r, %c2 : tensor<1x8x?xf16>
  return %h : index
}

// CHECK-LABEL: func.func @bnsh_value_hidden_product
// CHECK-SAME: %[[V:[^,]+]]: tensor<1x8x16x?xf16>
// CHECK: %[[D:.*]] = tensor.dim %[[V]], %{{.*}}
// CHECK: %[[H:.*]] = arith.muli %[[D]], %{{.*}} : index
// CHECK: return %[[H]] : index
func.func @bnsh_value_hidden_product(
    %ctx: !hip.context, %q: tensor<1x8x128xf16>,
    %k: tensor<1x8x16x16xf16>, %v: tensor<1x8x16x?xf16>,
    %out: tensor<1x8x?xf16>) -> index {
  %r = hip.multi_head_attention(%ctx)
      ins(%q, %k, %v : tensor<1x8x128xf16>, tensor<1x8x16x16xf16>, tensor<1x8x16x?xf16>)
      outs(%out : tensor<1x8x?xf16>) {num_heads = 8 : i64} : tensor<1x8x?xf16>
  %c2 = arith.constant 2 : index
  %h = tensor.dim %r, %c2 : tensor<1x8x?xf16>
  return %h : index
}

// CHECK-LABEL: func.func @optional_destination_extents
// CHECK-SAME: %[[PK:[^,]+]]: tensor<1x8x?x16xf16>
// CHECK-SAME: %[[QK:[^,]+]]: tensor<1x8x8x?xf32>
// CHECK: %[[CAP:.*]] = tensor.dim %[[PK]], %{{.*}}
// CHECK: %[[LEN:.*]] = tensor.dim %[[QK]], %{{.*}}
// CHECK: return %[[CAP]], %[[LEN]] : index, index
func.func @optional_destination_extents(
    %ctx: !hip.context, %q: tensor<1x8x128xf16>,
    %k: tensor<1x?x128xf16>, %v: tensor<1x?x64xf16>,
    %out: tensor<1x8x64xf16>, %pk: tensor<1x8x?x16xf16>,
    %qk: tensor<1x8x8x?xf32>) -> (index, index) {
  %r:3 = "hip.multi_head_attention"(%ctx, %q, %k, %v, %out, %pk, %qk) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1>,
      num_heads = 8 : i64
    } : (!hip.context, tensor<1x8x128xf16>, tensor<1x?x128xf16>, tensor<1x?x64xf16>, tensor<1x8x64xf16>, tensor<1x8x?x16xf16>, tensor<1x8x8x?xf32>) -> (tensor<1x8x64xf16>, tensor<1x8x?x16xf16>, tensor<1x8x8x?xf32>)
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %capacity = tensor.dim %r#1, %c2 : tensor<1x8x?x16xf16>
  %length = tensor.dim %r#2, %c3 : tensor<1x8x8x?xf32>
  return %capacity, %length : index, index
}
