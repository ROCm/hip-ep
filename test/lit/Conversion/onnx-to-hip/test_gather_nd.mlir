// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // batch_dims = 0 is default and elided from attr-dict.
  // data: 2x2 f32, indices: 2x2 i64 (each 2-tuple indexes both data dims)
  // -> output rank = q + r - indices_inner - 1 - batch_dims = 2 + 2 - 2 - 1 - 0 = 1
  func.func @gather_nd_basic(%data: tensor<2x2xf32>, %indices: tensor<2x2xi64>) -> tensor<2xf32> {
    %r = "onnx.GatherND"(%data, %indices) : (tensor<2x2xf32>, tensor<2x2xi64>) -> tensor<2xf32>
    return %r : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @gather_nd_basic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<2x2xf32>, %[[I:.*]]: tensor<2x2xi64>)
  // CHECK: tensor.empty() : tensor<2xf32>
  // CHECK: hip.gather_nd(%[[CTX]]) ins(%[[D]], %[[I]] : tensor<2x2xf32>, tensor<2x2xi64>) outs({{.*}} : tensor<2xf32>)

  // GatherND with batch_dims = 1 (non-default, so kept in attr-dict).
  func.func @gather_nd_batch1(%data: tensor<2x3x4xf32>, %indices: tensor<2x2x1xi64>) -> tensor<2x2x4xf32> {
    %r = "onnx.GatherND"(%data, %indices) {batch_dims = 1 : si64} : (tensor<2x3x4xf32>, tensor<2x2x1xi64>) -> tensor<2x2x4xf32>
    return %r : tensor<2x2x4xf32>
  }

  // CHECK-LABEL: func.func @gather_nd_batch1
  // CHECK: hip.gather_nd({{.*}}) ins({{.*}}, {{.*}} : tensor<2x3x4xf32>, tensor<2x2x1xi64>) outs({{.*}} : tensor<2x2x4xf32>) {batch_dims = 1 : i64}

  // Dynamic indices outer dim: indices.shape = [?, 2], data.shape =
  // [2, 2] -> output rank = q + r - k - 1 - batch_dims = 2 + 2 - 2 -
  // 1 - 0 = 1 with shape [?] sourced from indices dim 0 (the indices-
  // outer region: i < q-1).
  func.func @gather_nd_dyn_indices(%data: tensor<2x2xf32>, %indices: tensor<?x2xi64>) -> tensor<?xf32> {
    %r = "onnx.GatherND"(%data, %indices) : (tensor<2x2xf32>, tensor<?x2xi64>) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }

  // CHECK-LABEL: func.func @gather_nd_dyn_indices
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<2x2xf32>, %[[I:.*]]: tensor<?x2xi64>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[DIM:.*]] = tensor.dim %[[I]], %[[C0]] : tensor<?x2xi64>
  // CHECK: tensor.empty(%[[DIM]]) : tensor<?xf32>
  // CHECK: hip.gather_nd({{.*}}) ins({{.*}}, {{.*}} : tensor<2x2xf32>, tensor<?x2xi64>) outs({{.*}} : tensor<?xf32>)

  // Dynamic data tail dim: indices.shape = [3, 1] consumes axis 0 of
  // data; data.shape = [4, ?] -> result.shape = [3, ?] where the
  // trailing ? comes from data dim 1 (the data-tail region: i >= q-1).
  func.func @gather_nd_dyn_data_tail(%data: tensor<4x?xf32>, %indices: tensor<3x1xi64>) -> tensor<3x?xf32> {
    %r = "onnx.GatherND"(%data, %indices) : (tensor<4x?xf32>, tensor<3x1xi64>) -> tensor<3x?xf32>
    return %r : tensor<3x?xf32>
  }

  // CHECK-LABEL: func.func @gather_nd_dyn_data_tail
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<4x?xf32>, %[[I:.*]]: tensor<3x1xi64>)
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[DIM:.*]] = tensor.dim %[[D]], %[[C1]] : tensor<4x?xf32>
  // CHECK: tensor.empty(%[[DIM]]) : tensor<3x?xf32>
  // CHECK: hip.gather_nd
}
