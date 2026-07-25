// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // LessOrEqual(A, B) decomposes into Not(Less(B, A)) (note swapped operands),
  // each lowered to its own hip.* op. The original onnx.LessOrEqual must
  // disappear.
  func.func @lessorequal_f32(%a: tensor<3x4xf32>, %b: tensor<3x4xf32>) -> tensor<3x4xi1> {
    %r = "onnx.LessOrEqual"(%a, %b) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1>
    return %r : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @lessorequal_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<3x4xf32>, %[[B:.*]]: tensor<3x4xf32>)
  // CHECK-NOT: onnx.LessOrEqual
  // CHECK: hip.less(%[[CTX]]) ins(%[[B]], %[[A]] : tensor<3x4xf32>, tensor<3x4xf32>) outs({{.*}} : tensor<3x4xi1>)
  // CHECK: hip.not(%[[CTX]]) ins({{.*}} : tensor<3x4xi1>) outs({{.*}} : tensor<3x4xi1>)

  func.func @lessorequal_i32(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi1> {
    %r = "onnx.LessOrEqual"(%a, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    return %r : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @lessorequal_i32
  // CHECK-NOT: onnx.LessOrEqual
  // CHECK: hip.less({{.*}}) ins({{.*}}, {{.*}} : tensor<4xi32>, tensor<4xi32>) outs({{.*}} : tensor<4xi1>)
  // CHECK: hip.not({{.*}}) ins({{.*}} : tensor<4xi1>) outs({{.*}} : tensor<4xi1>)

  func.func @lessorequal_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xi1> {
    %r = "onnx.LessOrEqual"(%a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
    return %r : tensor<?x?xi1>
  }

  // CHECK-LABEL: func.func @lessorequal_dynamic
  // CHECK-NOT: onnx.LessOrEqual
  // CHECK: tensor.empty
  // CHECK: hip.less({{.*}}) ins({{.*}}, {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<?x?xi1>)
  // CHECK: hip.not({{.*}}) ins({{.*}} : tensor<?x?xi1>) outs({{.*}} : tensor<?x?xi1>)
}
