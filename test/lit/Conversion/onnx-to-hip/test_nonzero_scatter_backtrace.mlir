// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify NonZero + Transpose + ScatterND pipeline: the ScatterND conversion
// defers until NonZero is converted, then backtraces through Transpose to
// find NonZero's count_buf and passes it to hip.scatter_nd with
// has_valid_count = true.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4x5xf32>) -> tensor<4x5xf32> {
    return %arg0 : tensor<4x5xf32>
  }

  // NonZero -> Transpose -> ScatterND: count_buf should flow through.
  func.func @nonzero_transpose_scatter(
      %mask: tensor<4x5xf32>,
      %data: tensor<4x5x8xf32>,
      %updates: tensor<?x8xf32>) -> tensor<4x5x8xf32> {
    %nz = "onnx.NonZero"(%mask) : (tensor<4x5xf32>) -> tensor<2x?xi64>
    %t = "onnx.Transpose"(%nz) {perm = [1, 0]} : (tensor<2x?xi64>) -> tensor<?x2xi64>
    %out = "onnx.ScatterND"(%data, %t, %updates)
        : (tensor<4x5x8xf32>, tensor<?x2xi64>, tensor<?x8xf32>) -> tensor<4x5x8xf32>
    return %out : tensor<4x5x8xf32>
  }

  // CHECK-LABEL: func.func @nonzero_transpose_scatter
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

  // NonZero produces indices + count_buf (DPS: result is separate from init)
  // CHECK: %[[NZ:.*]]:2 = hip.nonzero(%[[CTX]]) ins({{.*}}) outs({{.*}} : {{.*}}, tensor<1xi32>)

  // Transpose on the indices (result #0)
  // CHECK: hip.transpose

  // ScatterND uses NonZero's count result (#1) with has_valid_count = true
  // CHECK: hip.scatter_nd(%[[CTX]]) ins({{.*}}, {{.*}}, {{.*}}, %[[NZ]]#1 :
  // CHECK-SAME: {has_valid_count = true}
}
