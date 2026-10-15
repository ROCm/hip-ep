// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX RandomNormalLike is lowered to hip.random_normal_like.
//
// The output copies the input shape. Dynamic dimensions stay dynamic and
// are read with tensor.dim. The input element type can differ from the
// output when dtype selects a float type.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128xi64>) -> tensor<1x128xi64> {
    return %arg0 : tensor<1x128xi64>
  }

  // All-dynamic rank 3. Default mean/scale are omitted from the print.
  func.func @test_rnl_dynamic_3d(%input: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
    // CHECK-LABEL: func.func @test_rnl_dynamic_3d
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
    %output = "onnx.RandomNormalLike"(%input) {seed = 1.0 : f32} : (tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[D0:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x?x?xf16>
    // CHECK-DAG: %[[D1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<?x?x?xf16>
    // CHECK-DAG: %[[D2:.*]] = tensor.dim %[[IN]], %[[C2]] : tensor<?x?x?xf16>
    // CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]]) : tensor<?x?x?xf16>
    // CHECK: hip.random_normal_like(%[[CTX]]) ins(%[[IN]] : tensor<?x?x?xf16>) outs(%[[INIT]] : tensor<?x?x?xf16>) {seed = 1.000000e+00 : f32}
    // CHECK-NOT: onnx.RandomNormalLike
    return %output : tensor<?x?x?xf16>
  }

  // Non-default mean and scale are kept. A static shape needs no tensor.dim.
  func.func @test_rnl_static_mean_scale(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @test_rnl_static_mean_scale
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<2x3xf32>) -> tensor<2x3xf32>
    %output = "onnx.RandomNormalLike"(%input) {mean = 2.0 : f32, scale = 0.5 : f32, seed = 3.0 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x3xf32>
    // CHECK: hip.random_normal_like(%[[CTX]]) ins(%[[IN]] : tensor<2x3xf32>) outs(%[[INIT]] : tensor<2x3xf32>) {mean = 2.000000e+00 : f32, scale = 5.000000e-01 : f32, seed = 3.000000e+00 : f32}
    // CHECK-NOT: onnx.RandomNormalLike
    return %output : tensor<2x3xf32>
  }

  // Integer input, float output: only the shape is copied.
  func.func @test_rnl_int_input(%input: tensor<4x8xi32>) -> tensor<4x8xf32> {
    // CHECK-LABEL: func.func @test_rnl_int_input
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<4x8xi32>) -> tensor<4x8xf32>
    %output = "onnx.RandomNormalLike"(%input) {dtype = 1 : si64, seed = 1.0 : f32} : (tensor<4x8xi32>) -> tensor<4x8xf32>
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x8xf32>
    // CHECK: hip.random_normal_like(%[[CTX]]) ins(%[[IN]] : tensor<4x8xi32>) outs(%[[INIT]] : tensor<4x8xf32>) {seed = 1.000000e+00 : f32}
    // CHECK-NOT: onnx.RandomNormalLike
    return %output : tensor<4x8xf32>
  }

  // Rank above 8 stays as the ONNX op.
  func.func @test_rnl_rank9(%input: tensor<1x1x1x1x1x1x1x1x1xf32>) -> tensor<1x1x1x1x1x1x1x1x1xf32> {
    // CHECK-LABEL: func.func @test_rnl_rank9
    // CHECK-NOT: hip.random_normal_like
    // CHECK: onnx.RandomNormalLike
    %output = "onnx.RandomNormalLike"(%input) {seed = 1.0 : f32} : (tensor<1x1x1x1x1x1x1x1x1xf32>) -> tensor<1x1x1x1x1x1x1x1x1xf32>
    return %output : tensor<1x1x1x1x1x1x1x1x1xf32>
  }
}
