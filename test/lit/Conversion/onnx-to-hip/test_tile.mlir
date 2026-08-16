// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @tile_f32(%input: tensor<2x3xf32>) -> tensor<4x9xf32> {
    %repeats = arith.constant dense<[2, 3]> : tensor<2xi64>
    %r = "onnx.Tile"(%input, %repeats)
      : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<4x9xf32>
    return %r : tensor<4x9xf32>
  }

  // CHECK-LABEL: func.func @tile_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<2x3xf32>)
  // CHECK: tensor.empty() : tensor<4x9xf32>
  // CHECK: hip.tile(%[[CTX]]) ins(%[[IN]], %{{.*}} : tensor<2x3xf32>, tensor<2xi64>) outs({{.*}} : tensor<4x9xf32>)
  // CHECK-SAME: static_repeats = array<i64: 2, 3>

  func.func @tile_dynamic(%input: tensor<?x?xf32>, %repeats: tensor<2xi64>) -> tensor<?x?xf32> {
    %r = "onnx.Tile"(%input, %repeats) : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @tile_dynamic
  // CHECK: %[[RB:.*]]:2 = hip.readback_shape
  // CHECK: %[[D0:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x?xf32>
  // CHECK: %[[D1:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x?xf32>
  // CHECK: %[[O0:.*]] = arith.muli %[[D0]], %[[RB]]#0 : index
  // CHECK: %[[O1:.*]] = arith.muli %[[D1]], %[[RB]]#1 : index
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[O0]], %[[O1]]) : tensor<?x?xf32>
  // CHECK: hip.tile({{.*}}) ins({{.*}}, {{.*}} : tensor<?x?xf32>, tensor<2xi64>) outs(%[[INIT]] : tensor<?x?xf32>)

  // Runtime repeats may coexist with imported static result extents. Only the
  // dynamic destination dimension needs payload-derived size arithmetic.
  func.func @tile_runtime_repeats_partial_static(
      %input: tensor<?x1152xf32>,
      %repeats: tensor<2xi64>) -> tensor<?x1152xf32> {
    %r = "onnx.Tile"(%input, %repeats)
      : (tensor<?x1152xf32>, tensor<2xi64>) -> tensor<?x1152xf32>
    return %r : tensor<?x1152xf32>
  }

  // CHECK-LABEL: func.func @tile_runtime_repeats_partial_static
  // CHECK: %[[PRB:.*]]:2 = hip.readback_shape
  // CHECK: %[[PD0:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x1152xf32>
  // CHECK: %[[PO0:.*]] = arith.muli %[[PD0]], %[[PRB]]#0 : index
  // CHECK: %[[PINIT:.*]] = tensor.empty(%[[PO0]]) : tensor<?x1152xf32>
  // CHECK: hip.tile
  // CHECK-SAME: outs(%[[PINIT]] : tensor<?x1152xf32>)

  func.func @tile_dynamic_input_constant_repeats(
      %input: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %repeats = arith.constant dense<[2, 3]> : tensor<2xi64>
    %r = "onnx.Tile"(%input, %repeats)
      : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @tile_dynamic_input_constant_repeats
  // CHECK-NOT: hip.readback_shape
  // CHECK: %[[CD0:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x?xf32>
  // CHECK: %[[CO0:.*]] = arith.muli %[[CD0]], %{{.*}} : index
  // CHECK: hip.tile
  // CHECK-SAME: static_repeats = array<i64: 2, 3>

  // Imported ONNX constants are dense hip.constant carriers before compute
  // conversion. Tile reads the carrier directly; no tile-specific stamp or
  // prepass is required.
  func.func @tile_carrier_repeats(
      %input: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %repeats = "onnx.Constant"() {
      value = dense<[4, 5]> : tensor<2xi64>
    } : () -> tensor<2xi64>
    %r = "onnx.Tile"(%input, %repeats)
      : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @tile_carrier_repeats
  // CHECK-NOT: hip.readback_shape
  // CHECK-NOT: hipdnn.tile_repeats
  // CHECK: hip.tile
  // CHECK-SAME: static_repeats = array<i64: 4, 5>

  func.func @tile_zero_repeat(%input: tensor<2x3xf32>) -> tensor<0x3xf32> {
    %repeats = arith.constant dense<[0, 1]> : tensor<2xi64>
    %r = "onnx.Tile"(%input, %repeats)
      : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<0x3xf32>
    return %r : tensor<0x3xf32>
  }

  // CHECK-LABEL: func.func @tile_zero_repeat
  // CHECK: tensor.empty() : tensor<0x3xf32>
  // CHECK: hip.tile
  // CHECK-SAME: static_repeats = array<i64: 0, 1>
}
