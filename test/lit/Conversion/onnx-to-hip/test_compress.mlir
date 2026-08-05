// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> {
    return %arg0 : tensor<3x2xf32>
  }

  func.func @compress_axis0(
      %arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<2x2xf32> {
    %result = "onnx.Compress"(%arg0, %arg1) {axis = 0 : si64}
        : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<2x2xf32>
    return %result : tensor<2x2xf32>
  }

  // CHECK-LABEL: func.func @compress_axis0
  // CHECK: hip.compress(%[[CTX:.*]]) ins(%[[IN:.*]], %[[COND:.*]] : tensor<3x2xf32>, tensor<3xi1>) outs({{.*}} : tensor<2x2xf32>) : tensor<2x2xf32>
  // CHECK-NOT: onnx.Compress

  func.func @compress_flatten(
      %arg0: tensor<2x2xf32>, %arg1: tensor<3xi1>) -> tensor<2xf32> {
    %result = "onnx.Compress"(%arg0, %arg1)
        : (tensor<2x2xf32>, tensor<3xi1>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @compress_flatten
  // CHECK: hip.compress(%{{.*}}) ins({{.*}} : tensor<2x2xf32>, tensor<3xi1>) outs({{.*}} : tensor<2xf32>) {flatten = true} : tensor<2xf32>

  // A dynamic selected extent is the number of TRUE condition entries, not the
  // condition length: scan it with hip.nonzero, read the count back to the
  // host, and size the destination (and hence any hip.alloc_output for a graph
  // output) with it.
  func.func @compress_axis0_dynamic(
      %arg0: tensor<?x2xf32>, %arg1: tensor<?xi1>) -> tensor<?x2xf32> {
    %result = "onnx.Compress"(%arg0, %arg1) {axis = 0 : si64}
        : (tensor<?x2xf32>, tensor<?xi1>) -> tensor<?x2xf32>
    return %result : tensor<?x2xf32>
  }

  // CHECK-LABEL: func.func @compress_axis0_dynamic
  // CHECK: %[[LEN:.*]] = tensor.dim %[[COND:.*]], %{{.*}} : tensor<?xi1>
  // CHECK: %[[IDX:.*]] = tensor.empty(%[[LEN]]) : tensor<1x?xi64>
  // CHECK: %[[CNT:.*]] = tensor.empty() : tensor<i32>
  // CHECK: %[[NZ:.*]]:2 = hip.nonzero(%[[CTX:.*]]) ins(%[[COND]] : tensor<?xi1>) outs(%[[IDX]], %[[CNT]] : tensor<1x?xi64>, tensor<i32>) {input_data_type = 5 : i64} : tensor<1x?xi64>, tensor<i32>
  // CHECK: %[[N:.*]] = hip.readback_dim(%[[CTX]], %[[NZ]]#1 : tensor<i32>) -> index
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[N]]) : tensor<?x2xf32>
  // CHECK: hip.compress(%[[CTX]]) ins(%{{.*}}, %[[COND]] : tensor<?x2xf32>, tensor<?xi1>) outs(%[[INIT]] : tensor<?x2xf32>) : tensor<?x2xf32>

  func.func @compress_flatten_dynamic(
      %arg0: tensor<2x2xf32>, %arg1: tensor<4xi1>) -> tensor<?xf32> {
    %result = "onnx.Compress"(%arg0, %arg1)
        : (tensor<2x2xf32>, tensor<4xi1>) -> tensor<?xf32>
    return %result : tensor<?xf32>
  }

  // CHECK-LABEL: func.func @compress_flatten_dynamic
  // CHECK: hip.nonzero(%[[CTX2:.*]]) ins(%[[COND2:.*]] : tensor<4xi1>)
  // CHECK: %[[N2:.*]] = hip.readback_dim(%[[CTX2]], %{{.*}} : tensor<i32>) -> index
  // CHECK: %[[INIT2:.*]] = tensor.empty(%[[N2]]) : tensor<?xf32>
  // CHECK: hip.compress(%[[CTX2]]) ins(%{{.*}}, %[[COND2]] : tensor<2x2xf32>, tensor<4xi1>) outs(%[[INIT2]] : tensor<?xf32>) {flatten = true} : tensor<?xf32>
}
