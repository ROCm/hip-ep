// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --test-hip-whole-shape-dim-reify %s | FileCheck %s

// CHECK-LABEL: func.func @dynamic_hidden
// CHECK-SAME: %[[Q:[^,]+]]: tensor<?x?x?xf16>
// CHECK-SAME: %[[K:[^,]+]]: tensor<?x?x?xf16>
// CHECK-SAME: %[[V:[^,]+]]: tensor<?x?x?xf16>
// CHECK: %[[VH0:.*]] = tensor.dim %[[V]], %{{.*}}
// CHECK: %[[DV0:.*]] = arith.divui %[[VH0]], %{{.*}} : index
// CHECK: %[[OH:.*]] = arith.muli %[[DV0]], %{{.*}} : index
// CHECK: %[[B:.*]] = tensor.dim %[[Q]], %{{.*}}
// CHECK: %[[QH:.*]] = tensor.dim %[[Q]], %{{.*}}
// CHECK: %[[DK:.*]] = arith.divui %[[QH]], %{{.*}} : index
// CHECK: %[[VH1:.*]] = tensor.dim %[[V]], %{{.*}}
// CHECK: %[[DV1:.*]] = arith.divui %[[VH1]], %{{.*}} : index
// CHECK: return %[[OH]], %[[B]], %{{.*}}, %[[DK]], %[[DV1]]
func.func @dynamic_hidden(
    %ctx: !hip.context,
    %query: tensor<?x?x?xf16>,
    %key: tensor<?x?x?xf16>,
    %value: tensor<?x?x?xf16>,
    %output: tensor<?x?x?xf16>,
    %state: tensor<?x8x?x?xf16>) -> (index, index, index, index, index) {
  %result:2 = hip.linear_attention(%ctx)
      ins(%query, %key, %value :
          tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>)
      outs(%output, %state : tensor<?x?x?xf16>, tensor<?x8x?x?xf16>)
      {q_num_heads = 32 : i64, kv_num_heads = 8 : i64}
      : tensor<?x?x?xf16>, tensor<?x8x?x?xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %out_hidden = tensor.dim %result#0, %c2 : tensor<?x?x?xf16>
  %state_batch = tensor.dim %result#1, %c0 : tensor<?x8x?x?xf16>
  %state_heads = tensor.dim %result#1, %c1 : tensor<?x8x?x?xf16>
  %state_dk = tensor.dim %result#1, %c2 : tensor<?x8x?x?xf16>
  %state_dv = tensor.dim %result#1, %c3 : tensor<?x8x?x?xf16>
  return %out_hidden, %state_batch, %state_heads, %state_dk, %state_dv
      : index, index, index, index, index
}

// -----

// CHECK-LABEL: func.func @static_hidden_i64_boundary
// CHECK: %[[BOUNDARY:.*]] = arith.constant 9223372036854775806 : index
// CHECK: return %[[BOUNDARY]] : index
func.func @static_hidden_i64_boundary(
    %ctx: !hip.context,
    %query: tensor<1x1x2xf16>,
    %key: tensor<1x1x1xf16>,
    %value: tensor<1x1x4611686018427387903xf16>,
    %output: tensor<1x1x9223372036854775806xf16>,
    %state: tensor<1x1x1x4611686018427387903xf16>) -> index {
  %result:2 = hip.linear_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x1x2xf16>, tensor<1x1x1xf16>,
          tensor<1x1x4611686018427387903xf16>)
      outs(%output, %state :
           tensor<1x1x9223372036854775806xf16>,
           tensor<1x1x1x4611686018427387903xf16>)
      {q_num_heads = 2 : i64, kv_num_heads = 1 : i64}
      : tensor<1x1x9223372036854775806xf16>,
        tensor<1x1x1x4611686018427387903xf16>
  %c2 = arith.constant 2 : index
  %hidden = tensor.dim %result#0, %c2
      : tensor<1x1x9223372036854775806xf16>
  return %hidden : index
}
