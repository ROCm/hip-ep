// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ReduceL2 is lowered DIRECTLY to the first-class hip.reduce_l2
// op. The sqrt(sum(x^2)) happens in-kernel, so the conversion is dim-tolerant.
//
// This test validates:
// - onnx.ReduceL2 -> hip.reduce_l2 (no Mul/ReduceSum/Sqrt decomposition)
// - keepdims = 1 and keepdims = 0 both supported
// - SwinV2 attention L2-norm shape: [128x3x256x32] -> [128x3x256x1]
// - Dynamic input shapes with keepdims = 0
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x4xf16>) -> tensor<1x4xf16> {
    return %arg0 : tensor<1x4xf16>
  }

  // SwinV2 Q/K L2 norm: trailing axis -1, keepdims=1.
  func.func @test_reduce_l2_swinv2_attn(%data: tensor<128x3x256x32xf16>, %axes: tensor<1xi64>) -> tensor<128x3x256x1xf16> {
    // CHECK-LABEL: func.func @test_reduce_l2_swinv2_attn
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<128x3x256x32xf16>, %[[AXES:.*]]: tensor<1xi64>) -> tensor<128x3x256x1xf16>

    %output = "onnx.ReduceL2"(%data, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<128x3x256x32xf16>, tensor<1xi64>) -> tensor<128x3x256x1xf16>

    // CHECK: tensor.empty() : tensor<128x3x256x1xf16>
    // CHECK: hip.reduce_l2(%[[CTX]]) ins(%[[DATA]], %[[AXES]] : tensor<128x3x256x32xf16>, tensor<1xi64>) outs({{.*}} : tensor<128x3x256x1xf16>)
    // CHECK-NOT: onnx.ReduceSum
    // CHECK-NOT: onnx.Sqrt
    // CHECK-NOT: hip.alloc

    return %output : tensor<128x3x256x1xf16>
  }

  func.func @test_reduce_l2_keepdims(%data: tensor<1x128xf16>, %axes: tensor<i64>) -> tensor<1x1xf16> {
    // CHECK-LABEL: func.func @test_reduce_l2_keepdims

    %output = "onnx.ReduceL2"(%data, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x128xf16>, tensor<i64>) -> tensor<1x1xf16>

    // CHECK: hip.reduce_l2(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1x128xf16>, tensor<i64>) outs({{.*}} : tensor<1x1xf16>)

    return %output : tensor<1x1xf16>
  }

  func.func @test_reduce_l2_no_keepdims(%data: tensor<4x8xf16>, %axes: tensor<i64>) -> tensor<4xf16> {
    // CHECK-LABEL: func.func @test_reduce_l2_no_keepdims

    %output = "onnx.ReduceL2"(%data, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<4x8xf16>, tensor<i64>) -> tensor<4xf16>

    // CHECK: hip.reduce_l2(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<4x8xf16>, tensor<i64>) outs({{.*}} : tensor<4xf16>) {keepdims = 0 : i64}

    return %output : tensor<4xf16>
  }

  func.func @reduce_l2_dynamic(%data: tensor<?x?x512xf16>, %axes: tensor<i64>) -> tensor<?x?xf16> {
    %output = "onnx.ReduceL2"(%data, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x?x512xf16>, tensor<i64>) -> tensor<?x?xf16>
    return %output : tensor<?x?xf16>
  }

  // CHECK-LABEL: func.func @reduce_l2_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<?x?x512xf16>, %[[AXES:.*]]: tensor<i64>) -> tensor<?x?xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xf16>
  // CHECK: hip.reduce_l2(%[[CTX]]) ins(%[[DATA]], %[[AXES]] : tensor<?x?x512xf16>, tensor<i64>) outs(%[[INIT]] : tensor<?x?xf16>) {keepdims = 0 : i64}
  // CHECK-NOT: hip.alloc
}
