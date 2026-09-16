// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @expand_static(%input: tensor<3x1xf32>, %shape: tensor<3xi64>) -> tensor<2x3x6xf32> {
    %r = "onnx.Expand"(%input, %shape) : (tensor<3x1xf32>, tensor<3xi64>) -> tensor<2x3x6xf32>
    return %r : tensor<2x3x6xf32>
  }

  // CHECK-LABEL: func.func @expand_static
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<3x1xf32>, %[[SH:.*]]: tensor<3xi64>)
  // CHECK: tensor.empty() : tensor<2x3x6xf32>
  // CHECK: hip.expand(%[[CTX]]) ins(%[[IN]], %[[SH]] : tensor<3x1xf32>, tensor<3xi64>) outs({{.*}} : tensor<2x3x6xf32>)

  func.func @expand_dynamic(%input: tensor<3x1xf32>, %shape: tensor<3xi64>) -> tensor<?x3x?xf32> {
    %r = "onnx.Expand"(%input, %shape) : (tensor<3x1xf32>, tensor<3xi64>) -> tensor<?x3x?xf32>
    return %r : tensor<?x3x?xf32>
  }

  // CHECK-LABEL: func.func @expand_dynamic
  // CHECK: tensor.extract
  // CHECK: tensor.extract
  // CHECK: tensor.empty
  // CHECK: hip.expand({{.*}}) ins({{.*}}, {{.*}} : tensor<3x1xf32>, tensor<3xi64>) outs({{.*}} : tensor<?x3x?xf32>)

  // --- Broadcast-1 must keep the input's extent, not resize to 1 ---
  // Result dim 0 is a pure rank-broadcast prefix with no input counterpart, so
  // its shape entry IS the literal extent.  Result dims 1 and 2 do have input
  // counterparts (extents 3 and 5), so a `1` there means "don't broadcast this
  // position" and the extent must come from the input.
  func.func @expand_broadcast_one(%input: tensor<3x5xf32>, %shape: tensor<3xi64>) -> tensor<?x?x?xf32> {
    %r = "onnx.Expand"(%input, %shape) : (tensor<3x5xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
    return %r : tensor<?x?x?xf32>
  }

  // CHECK-LABEL: func.func @expand_broadcast_one
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // No input dim at result index 0: entry used verbatim, nothing to compare.
  // CHECK: %[[E0:.*]] = hip.readback_scalar
  // CHECK-NEXT: %[[D0:.*]] = arith.index_cast %[[E0]] : i64 to index
  // CHECK: %[[E1:.*]] = hip.readback_scalar
  // CHECK-NEXT: %[[S1:.*]] = arith.index_cast %[[E1]] : i64 to index
  // CHECK-NEXT: %[[I0:.*]] = tensor.dim %{{.*}}, %[[C0]] : tensor<3x5xf32>
  // CHECK-NEXT: %[[ONE1:.*]] = arith.cmpi eq, %[[S1]], %[[C1]] : index
  // CHECK-NEXT: %[[D1:.*]] = arith.select %[[ONE1]], %[[I0]], %[[S1]] : index
  // CHECK: %[[E2:.*]] = hip.readback_scalar
  // CHECK-NEXT: %[[S2:.*]] = arith.index_cast %[[E2]] : i64 to index
  // CHECK-NEXT: %[[I1:.*]] = tensor.dim %{{.*}}, %[[C1]] : tensor<3x5xf32>
  // CHECK-NEXT: %[[ONE2:.*]] = arith.cmpi eq, %[[S2]], %[[C1]] : index
  // CHECK-NEXT: %[[D2:.*]] = arith.select %[[ONE2]], %[[I1]], %[[S2]] : index
  // Each dynamic extent reaches the init tensor as the value resolved above.
  // CHECK: tensor.empty(%[[D0]], %[[D1]], %[[D2]]) : tensor<?x?x?xf32>
  // CHECK: hip.expand
}
