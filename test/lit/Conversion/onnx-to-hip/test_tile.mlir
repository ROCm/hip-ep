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

  // Constant repeats of unit extents only: a plain broadcast, no reshapes.
  func.func @tile_unit_dims(%input: tensor<1x4x1x8xf32>) -> tensor<1x4x3x8xf32> {
    %repeats = "onnx.Constant"() {value = dense<[1, 1, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %r = "onnx.Tile"(%input, %repeats) : (tensor<1x4x1x8xf32>, tensor<4xi64>) -> tensor<1x4x3x8xf32>
    return %r : tensor<1x4x3x8xf32>
  }

  // CHECK-LABEL: func.func @tile_unit_dims
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x4x1x8xf32>)
  // CHECK-NOT: tensor.expand_shape
  // CHECK: %[[SHAPE:.*]] = hip.constant {{.*}}dense<[1, 4, 3, 8]> : tensor<4xi64>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x4x3x8xf32>
  // CHECK: %[[B:.*]] = hip.expand(%[[CTX]]) ins(%[[IN]], %[[SHAPE]] : tensor<1x4x1x8xf32>, tensor<4xi64>) outs(%[[INIT]] : tensor<1x4x3x8xf32>)
  // CHECK-NOT: tensor.collapse_shape
  // CHECK-NOT: hip.tile
  // CHECK: return %[[B]]

  // A repeated extent above one gets its own unit dim to broadcast, then
  // merges back with it; the repeated unit extent broadcasts in place.
  func.func @tile_general(%input: tensor<2x1x3xf32>) -> tensor<4x4x3xf32> {
    %repeats = "onnx.Constant"() {value = dense<[2, 4, 1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %r = "onnx.Tile"(%input, %repeats) : (tensor<2x1x3xf32>, tensor<3xi64>) -> tensor<4x4x3xf32>
    return %r : tensor<4x4x3xf32>
  }

  // CHECK-LABEL: func.func @tile_general
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<2x1x3xf32>)
  // CHECK: %[[E:.*]] = tensor.expand_shape %[[IN]] {{\[}}[0, 1], [2], [3]] output_shape [1, 2, 1, 3] : tensor<2x1x3xf32> into tensor<1x2x1x3xf32>
  // CHECK: %[[SHAPE:.*]] = hip.constant {{.*}}dense<[2, 2, 4, 3]> : tensor<4xi64>
  // CHECK: %[[B:.*]] = hip.expand(%[[CTX]]) ins(%[[E]], %[[SHAPE]] : tensor<1x2x1x3xf32>, tensor<4xi64>) outs({{.*}} : tensor<2x2x4x3xf32>)
  // CHECK: %[[Y:.*]] = tensor.collapse_shape %[[B]] {{\[}}[0, 1], [2], [3]] : tensor<2x2x4x3xf32> into tensor<4x4x3xf32>
  // CHECK-NOT: hip.tile
  // CHECK: return %[[Y]]

  // A zero repeat empties the result; keep hip.tile.
  func.func @tile_zero_repeat(%input: tensor<2x3xf32>) -> tensor<0x3xf32> {
    %repeats = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
    %r = "onnx.Tile"(%input, %repeats) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<0x3xf32>
    return %r : tensor<0x3xf32>
  }

  // CHECK-LABEL: func.func @tile_zero_repeat
  // CHECK-NOT: hip.expand
  // CHECK: hip.tile

  // Dynamic extents with constant repeats; keep hip.tile.
  func.func @tile_dynamic_const_repeats(%input: tensor<?x3xf32>) -> tensor<?x6xf32> {
    %repeats = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %r = "onnx.Tile"(%input, %repeats) : (tensor<?x3xf32>, tensor<2xi64>) -> tensor<?x6xf32>
    return %r : tensor<?x6xf32>
  }

  // CHECK-LABEL: func.func @tile_dynamic_const_repeats
  // CHECK-NOT: hip.expand
  // CHECK: hip.tile

  // Every extent and repeat above one doubles the rank to 10, past what the
  // expand kernel takes; keep hip.tile.
  func.func @tile_rank_cap(%input: tensor<2x2x2x2x2xf32>) -> tensor<4x4x4x4x4xf32> {
    %repeats = "onnx.Constant"() {value = dense<2> : tensor<5xi64>} : () -> tensor<5xi64>
    %r = "onnx.Tile"(%input, %repeats) : (tensor<2x2x2x2x2xf32>, tensor<5xi64>) -> tensor<4x4x4x4x4xf32>
    return %r : tensor<4x4x4x4x4xf32>
  }

  // CHECK-LABEL: func.func @tile_rank_cap
  // CHECK-NOT: hip.expand
  // CHECK: hip.tile
}
