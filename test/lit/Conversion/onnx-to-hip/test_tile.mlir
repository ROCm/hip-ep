// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @tile_f32(%input: tensor<2x3xf32>, %repeats: tensor<2xi64>) -> tensor<4x9xf32> {
    %r = "onnx.Tile"(%input, %repeats) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<4x9xf32>
    return %r : tensor<4x9xf32>
  }

  // CHECK-LABEL: func.func @tile_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<2x3xf32>, %[[R:.*]]: tensor<2xi64>)
  // CHECK: tensor.empty() : tensor<4x9xf32>
  // CHECK: hip.tile(%[[CTX]]) ins(%[[IN]], %[[R]] : tensor<2x3xf32>, tensor<2xi64>) outs({{.*}} : tensor<4x9xf32>)

  func.func @tile_dynamic(%input: tensor<?x?xf32>, %repeats: tensor<2xi64>) -> tensor<?x?xf32> {
    %r = "onnx.Tile"(%input, %repeats) : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @tile_dynamic
  // CHECK: hip.tile({{.*}}) ins({{.*}}, {{.*}} : tensor<?x?xf32>, tensor<2xi64>) outs({{.*}} : tensor<?x?xf32>)
}
