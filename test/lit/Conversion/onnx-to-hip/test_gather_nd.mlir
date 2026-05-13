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
}
