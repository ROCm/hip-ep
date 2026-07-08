// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
    return %arg0 : tensor<3x4xf32>
  }

  func.func @topk_axis1_static(
      %x: tensor<3x4xf32>, %k: tensor<i64>) -> (tensor<3x2xf32>, tensor<3x2xi64>) {
    %values, %indices = "onnx.TopK"(%x, %k) {
      axis = 1 : si64,
      largest = 1 : si64,
      sorted = 1 : si64
    } : (tensor<3x4xf32>, tensor<i64>) -> (tensor<3x2xf32>, tensor<3x2xi64>)
    return %values, %indices : tensor<3x2xf32>, tensor<3x2xi64>
  }

  // CHECK-LABEL: func.func @topk_axis1_static
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<3x4xf32>, %[[K:.*]]: tensor<i64>)
  // CHECK: tensor.empty() : tensor<3x2xf32>
  // CHECK: tensor.empty() : tensor<3x2xi64>
  // CHECK: hip.top_k(%[[CTX]]) ins(%[[X]], %[[K]] : tensor<3x4xf32>, tensor<i64>) outs({{.*}}, {{.*}} : tensor<3x2xf32>, tensor<3x2xi64>) {axis = 1 : i64} : tensor<3x2xf32>, tensor<3x2xi64>
  // CHECK-NOT: onnx.TopK

  func.func @topk_last_axis_f16(
      %x: tensor<2x5xf16>, %k: tensor<1xi64>) -> (tensor<2x3xf16>, tensor<2x3xi64>) {
    %values, %indices = "onnx.TopK"(%x, %k) {axis = -1 : si64}
        : (tensor<2x5xf16>, tensor<1xi64>) -> (tensor<2x3xf16>, tensor<2x3xi64>)
    return %values, %indices : tensor<2x3xf16>, tensor<2x3xi64>
  }

  // CHECK-LABEL: func.func @topk_last_axis_f16
  // CHECK: hip.top_k(%{{.*}}) ins({{.*}} : tensor<2x5xf16>, tensor<1xi64>) outs({{.*}}, {{.*}} : tensor<2x3xf16>, tensor<2x3xi64>) {axis = 1 : i64} : tensor<2x3xf16>, tensor<2x3xi64>

  func.func @topk_dynamic(
      %x: tensor<?x?xf32>, %k: tensor<1xi64>) -> (tensor<?x?xf32>, tensor<?x?xi64>) {
    %values, %indices = "onnx.TopK"(%x, %k) {axis = 1 : si64}
        : (tensor<?x?xf32>, tensor<1xi64>) -> (tensor<?x?xf32>, tensor<?x?xi64>)
    return %values, %indices : tensor<?x?xf32>, tensor<?x?xi64>
  }

  // CHECK-LABEL: func.func @topk_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[X3:.*]]: tensor<?x?xf32>, %[[K3:.*]]: tensor<1xi64>)
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = tensor.dim %[[X3]], %[[C0]] : tensor<?x?xf32>
  // CHECK: hip.readback_scalar(%[[CTX3]], %{{.*}} : tensor<1xi64>) -> i64
  // CHECK: tensor.empty({{.*}}, {{.*}}) : tensor<?x?xf32>
  // CHECK: tensor.empty({{.*}}, {{.*}}) : tensor<?x?xi64>
  // CHECK: hip.top_k(%[[CTX3]]) ins(%[[X3]], %[[K3]] : tensor<?x?xf32>, tensor<1xi64>) outs({{.*}}, {{.*}} : tensor<?x?xf32>, tensor<?x?xi64>) {axis = 1 : i64} : tensor<?x?xf32>, tensor<?x?xi64>
}
