// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --canonicalize --cse --verify-each | FileCheck %s --implicit-check-not=hip.readback_scalar --implicit-check-not=tensor.extract

// Exercise producer conversion followed by the ordinary canonicalizer, not a
// hand-built host shortcut. Logical length and past capacity remain distinct.
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: %[[PAST:[^,]+]]: tensor<1x2x?x4xf16>
// CHECK-SAME: %[[MASK:[^,]+]]: tensor<1x?xi64>
// CHECK: %[[DIM:.*]] = tensor.dim %[[MASK]]
// CHECK: %[[I64:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK: %[[PACKED:.*]] = tensor.from_elements %[[I64]] : tensor<i64>
// CHECK: %[[GPU:.*]] = hip.cast{{.*}} ins(%[[PACKED]] : tensor<i64>)
// CHECK: %[[I32:.*]] = arith.trunci %[[I64]] : i64 to i32
// CHECK: %[[LOGICAL:.*]] = arith.index_cast %[[I32]] : i32 to index
// CHECK: %[[CAP:.*]] = tensor.dim %[[PAST]]
// CHECK: %[[MAX:.*]] = arith.maxui %[[CAP]], %[[LOGICAL]] : index
// CHECK: tensor.empty(%[[MAX]]) : tensor<1x2x?x4xf16>
// CHECK: hip.gqa
// CHECK-SAME: %[[GPU]]
func.func @main_graph(
    %query: tensor<1x1x16xf16>, %key: tensor<1x1x8xf16>,
    %value: tensor<1x1x8xf16>, %past: tensor<1x2x?x4xf16>,
    %mask: tensor<1x?xi64>, %seqlens: tensor<1xi32>)
    -> (tensor<1x1x16xf16>, tensor<1x2x?x4xf16>, tensor<1x2x?x4xf16>) {
  %shape = "onnx.Shape"(%mask) : (tensor<1x?xi64>) -> tensor<2xi64>
  %idx = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %dim = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
      : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
  %total = "onnx.Cast"(%dim) {to = i32} : (tensor<i64>) -> tensor<i32>
  %out:3 = "onnx.Custom"(%query, %key, %value, %past, %past, %seqlens, %total)
      {domain_name = "com.microsoft", function_name = "GroupQueryAttention",
       kv_num_heads = 2 : si64, num_heads = 4 : si64}
      : (tensor<1x1x16xf16>, tensor<1x1x8xf16>, tensor<1x1x8xf16>,
         tensor<1x2x?x4xf16>, tensor<1x2x?x4xf16>, tensor<1xi32>, tensor<i32>)
      -> (tensor<1x1x16xf16>, tensor<1x2x?x4xf16>, tensor<1x2x?x4xf16>)
  return %out#0, %out#1, %out#2
      : tensor<1x1x16xf16>, tensor<1x2x?x4xf16>, tensor<1x2x?x4xf16>
}
