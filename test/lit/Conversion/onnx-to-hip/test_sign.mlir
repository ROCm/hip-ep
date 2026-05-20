// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @sign_f32(%input: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %r = "onnx.Sign"(%input) : (tensor<3x4xf32>) -> tensor<3x4xf32>
    return %r : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @sign_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<3x4xf32>)
  // CHECK: tensor.empty() : tensor<3x4xf32>
  // CHECK: hip.sign(%[[CTX]]) ins(%[[IN]] : tensor<3x4xf32>) outs({{.*}} : tensor<3x4xf32>)

  func.func @sign_i32(%input: tensor<128xi32>) -> tensor<128xi32> {
    %r = "onnx.Sign"(%input) : (tensor<128xi32>) -> tensor<128xi32>
    return %r : tensor<128xi32>
  }

  // CHECK-LABEL: func.func @sign_i32
  // CHECK: hip.sign({{.*}}) ins({{.*}} : tensor<128xi32>) outs({{.*}} : tensor<128xi32>)

  func.func @sign_dynamic(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %r = "onnx.Sign"(%input) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @sign_dynamic
  // CHECK: hip.sign({{.*}}) ins({{.*}} : tensor<?x?xf32>) outs({{.*}} : tensor<?x?xf32>)
}
