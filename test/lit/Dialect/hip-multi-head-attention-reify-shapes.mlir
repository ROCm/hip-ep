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
