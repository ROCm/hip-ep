// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Default fmod = 0 -> integer modulo (Python-style). Default attr is
  // elided from attr-dict.
  func.func @mod_i32(%a: tensor<3x4xi32>, %b: tensor<3x4xi32>) -> tensor<3x4xi32> {
    %r = "onnx.Mod"(%a, %b) : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>
    return %r : tensor<3x4xi32>
  }

  // CHECK-LABEL: func.func @mod_i32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<3x4xi32>, %[[B:.*]]: tensor<3x4xi32>)
  // CHECK: tensor.empty() : tensor<3x4xi32>
  // CHECK: hip.mod(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<3x4xi32>, tensor<3x4xi32>) outs({{.*}} : tensor<3x4xi32>)

  // fmod = 1 -> C fmod (float). Non-default, so it stays in the attr-dict.
  func.func @mod_f32_fmod(%a: tensor<3x4xf32>, %b: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %r = "onnx.Mod"(%a, %b) {fmod = 1 : si64} : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %r : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @mod_f32_fmod
  // CHECK: hip.mod({{.*}}) ins({{.*}}, {{.*}} : tensor<3x4xf32>, tensor<3x4xf32>) outs({{.*}} : tensor<3x4xf32>) {fmod = 1 : i64}

  func.func @mod_dynamic(%a: tensor<?x?xi32>, %b: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %r = "onnx.Mod"(%a, %b) : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    return %r : tensor<?x?xi32>
  }

  // CHECK-LABEL: func.func @mod_dynamic
  // CHECK: hip.mod({{.*}}) ins({{.*}}, {{.*}} : tensor<?x?xi32>, tensor<?x?xi32>) outs({{.*}} : tensor<?x?xi32>)

  func.func @mod_rank5_broadcast_fmod(%a: tensor<2x3x4x5x6xf32>, %b: tensor<1x1x1x1x6xf32>) -> tensor<2x3x4x5x6xf32> {
    %r = "onnx.Mod"(%a, %b) {fmod = 1 : si64} : (tensor<2x3x4x5x6xf32>, tensor<1x1x1x1x6xf32>) -> tensor<2x3x4x5x6xf32>
    return %r : tensor<2x3x4x5x6xf32>
  }

  // CHECK-LABEL: func.func @mod_rank5_broadcast_fmod
  // CHECK: %[[PA:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0], [1], [2, 3], [4]] : tensor<2x3x4x5x6xf32> into tensor<2x3x20x6xf32>
  // CHECK: %[[PB:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0], [1], [2, 3], [4]] : tensor<1x1x1x1x6xf32> into tensor<1x1x1x6xf32>
  // CHECK: %[[R:.*]] = hip.mod({{.*}}) ins(%[[PA]], %[[PB]] : tensor<2x3x20x6xf32>, tensor<1x1x1x6xf32>) outs({{.*}} : tensor<2x3x20x6xf32>) {fmod = 1 : i64}
  // CHECK: tensor.expand_shape %[[R]] {{\[}}[0], [1], [2, 3], [4]] output_shape [2, 3, 4, 5, 6]
}
