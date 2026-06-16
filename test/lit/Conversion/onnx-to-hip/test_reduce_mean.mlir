// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ReduceMean is lowered DIRECTLY to the first-class hip.reduce_mean
// op (no ReduceSum + Div decomposition). The mean division happens in-kernel,
// so the conversion is dim-tolerant: it must fire even when the reduce axis is
// dynamic (the old ReduceMeanToReduceSumDiv decomposition bailed on a dynamic
// reduce dim).
//
// This test validates:
// - onnx.ReduceMean -> hip.reduce_mean (no onnx.ReduceSum, no Mul/Div left)
// - keepdims = 1 and keepdims = 0 both supported
// - tensor-first DPS: tensor.empty() used as output init
// - dynamic shapes (including a dynamic reduce axis) convert cleanly
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<1x4xf16>) -> tensor<1x4xf16> {
    return %arg0 : tensor<1x4xf16>
  }

  // keepdims = 1: reduced dim kept as size 1.
  func.func @test_reduce_mean_keepdims(%data: tensor<1x128xf16>, %axes: tensor<i64>) -> tensor<1x1xf16> {
    // CHECK-LABEL: func.func @test_reduce_mean_keepdims
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<1x128xf16>, %[[AXES:.*]]: tensor<i64>) -> tensor<1x1xf16>

    %output = "onnx.ReduceMean"(%data, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x128xf16>, tensor<i64>) -> tensor<1x1xf16>

    // CHECK: tensor.empty() : tensor<1x1xf16>
    // CHECK: hip.reduce_mean(%[[CTX]]) ins(%[[DATA]], %[[AXES]] : tensor<1x128xf16>, tensor<i64>) outs({{.*}} : tensor<1x1xf16>)
    // CHECK-NOT: onnx.ReduceSum
    // CHECK-NOT: hip.alloc

    return %output : tensor<1x1xf16>
  }

  // keepdims = 0: reduced dim dropped.
  func.func @test_reduce_mean_no_keepdims(%data: tensor<4x8xf16>, %axes: tensor<i64>) -> tensor<4xf16> {
    // CHECK-LABEL: func.func @test_reduce_mean_no_keepdims
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<4x8xf16>, %[[AXES:.*]]: tensor<i64>) -> tensor<4xf16>

    %output = "onnx.ReduceMean"(%data, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<4x8xf16>, tensor<i64>) -> tensor<4xf16>

    // CHECK: tensor.empty() : tensor<4xf16>
    // CHECK: hip.reduce_mean(%[[CTX]]) ins(%[[DATA]], %[[AXES]] : tensor<4x8xf16>, tensor<i64>) outs({{.*}} : tensor<4xf16>) {keepdims = 0 : i64}
    // CHECK-NOT: onnx.ReduceSum

    return %output : tensor<4xf16>
  }

  // Dynamic shape with a dynamic reduce axis: must STILL convert (the in-kernel
  // division means no static reduce dim is required).
  func.func @reduce_mean_dynamic(%data: tensor<?x?x512xf16>, %axes: tensor<i64>) -> tensor<?x?xf16> {
    %output = "onnx.ReduceMean"(%data, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x?x512xf16>, tensor<i64>) -> tensor<?x?xf16>
    return %output : tensor<?x?xf16>
  }

  // CHECK-LABEL: func.func @reduce_mean_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<?x?x512xf16>, %[[AXES:.*]]: tensor<i64>) -> tensor<?x?xf16>
  // CHECK: hip.reduce_mean(%[[CTX]]) ins(%[[DATA]], %[[AXES]] : tensor<?x?x512xf16>, tensor<i64>) outs({{.*}} : tensor<?x?xf16>) {keepdims = 0 : i64}
  // CHECK-NOT: onnx.ReduceSum
}
