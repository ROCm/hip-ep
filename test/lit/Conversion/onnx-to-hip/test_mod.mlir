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
}
