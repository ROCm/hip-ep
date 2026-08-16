// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ReduceSum is correctly lowered to hip.reduce_sum operation
// in tensor-first mode.
//
// This test validates:
// - Reduction operation lowering (onnx.ReduceSum -> hip.reduce_sum)
// - keepdims = 1: output shape keeps reduced dimension as size 1
// - keepdims = 0: output shape drops the reduced dimension entirely
// - i64 element type support
// - 2D tensor reduction with axes as input operand
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
//
// Model: Llama-3.1-8B attention mask sum computation
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<1x128xi64>) -> tensor<1x128xi64> {
    return %arg0 : tensor<1x128xi64>
  }

  // keepdims = 1: reduced dim is kept as size 1
  func.func @test_reduce_sum_keepdims(%data: tensor<1x128xi64>) -> tensor<1x1xi64> {
    // After conversion: context prepended, tensors remain tensors, tensor return
    // CHECK-LABEL: func.func @test_reduce_sum_keepdims
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<1x128xi64>) -> tensor<1x1xi64>

    %axes = arith.constant dense<1> : tensor<i64>
    %output = "onnx.ReduceSum"(%data, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x128xi64>, tensor<i64>) -> tensor<1x1xi64>

    // After conversion: tensor.empty() for init, hip.reduce_sum in tensor mode
    // CHECK: %[[AXES:.*]] = arith.constant dense<1> : tensor<1xi64>
    // CHECK: tensor.empty() : tensor<1x1xi64>
    // CHECK: hip.reduce_sum(%[[CTX]]) ins(%[[DATA]], %[[AXES]] : tensor<1x128xi64>, tensor<1xi64>) outs({{.*}} : tensor<1x1xi64>)
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<1x1xi64>
  }

  // keepdims = 0: reduced dim is dropped from output shape
  func.func @test_reduce_sum_no_keepdims(%data: tensor<4x8xi64>) -> tensor<4xi64> {
    // CHECK-LABEL: func.func @test_reduce_sum_no_keepdims
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<4x8xi64>) -> tensor<4xi64>

    %axes = arith.constant dense<[1]> : tensor<1xi64>
    %output = "onnx.ReduceSum"(%data, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<4x8xi64>, tensor<1xi64>) -> tensor<4xi64>

    // CHECK: %[[AXES:.*]] = arith.constant dense<1> : tensor<1xi64>
    // CHECK: tensor.empty() : tensor<4xi64>
    // CHECK: hip.reduce_sum(%[[CTX]]) ins(%[[DATA]], %[[AXES]] : tensor<4x8xi64>, tensor<1xi64>) outs({{.*}} : tensor<4xi64>) {keepdims = 0 : i64, normalized_axes = array<i64: 1>}
    // CHECK-NOT: hip.alloc

    return %output : tensor<4xi64>
  }

  // keepdims = 0 with compile-time axes and dynamic extents. Dropping the
  // reduced axes makes the output dimension order non-positional in the input:
  // output dim 1 must come from `data` dim 3, not dim 1. The destination is
  // built from the same shape helper that backs `reifyResultShapes`, so both
  // agree on that mapping.
  func.func @reduce_sum_no_keepdims_static_axes(%data: tensor<?x?x?x?xf32>)
      -> tensor<?x?xf32> {
    %output = "onnx.ReduceSum"(%data)
        {axes = [1 : si64, 2 : si64], keepdims = 0 : si64,
         noop_with_empty_axes = 0 : si64}
        : (tensor<?x?x?x?xf32>) -> tensor<?x?xf32>
    return %output : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @reduce_sum_no_keepdims_static_axes
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<?x?x?x?xf32>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[D0:.*]] = tensor.dim %[[DATA]], %[[C0]] : tensor<?x?x?x?xf32>
  // CHECK: %[[D3:.*]] = tensor.dim %[[DATA]], %[[C3]] : tensor<?x?x?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D3]]) : tensor<?x?xf32>
  // CHECK: hip.reduce_sum(%[[CTX]]) ins(%[[DATA]], %{{.*}} : tensor<?x?x?x?xf32>, tensor<2xi64>) outs(%[[INIT]] : tensor<?x?xf32>) {keepdims = 0 : i64, normalized_axes = array<i64: 1, 2>}

  // Same mapping when the axes arrive as a compile-time-constant OPERAND
  // (opset 13+) rather than an attribute. Gating destination construction on the
  // operand count instead of on the operand being constant would build a
  // positional destination here while reification computed the real mapping.
  func.func @reduce_sum_no_keepdims_constant_axes_operand(
      %data: tensor<?x?x?x?xf32>) -> tensor<?x?xf32> {
    %axes = arith.constant dense<[1, 2]> : tensor<2xi64>
    %output = "onnx.ReduceSum"(%data, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<?x?x?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    return %output : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @reduce_sum_no_keepdims_constant_axes_operand
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<?x?x?x?xf32>)
  // CHECK-DAG: %[[OC0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[OC3:.*]] = arith.constant 3 : index
  // CHECK: %[[OD0:.*]] = tensor.dim %[[DATA]], %[[OC0]] : tensor<?x?x?x?xf32>
  // CHECK: %[[OD3:.*]] = tensor.dim %[[DATA]], %[[OC3]] : tensor<?x?x?x?xf32>
  // CHECK: %[[OINIT:.*]] = tensor.empty(%[[OD0]], %[[OD3]]) : tensor<?x?xf32>
  // CHECK: hip.reduce_sum(%[[CTX]]) ins(%[[DATA]], %{{.*}} : tensor<?x?x?x?xf32>, tensor<2xi64>) outs(%[[OINIT]] : tensor<?x?xf32>) {keepdims = 0 : i64, normalized_axes = array<i64: 1, 2>}

  // Empty constant axes reduce every dimension when noop_with_empty_axes=0.
  func.func @reduce_sum_empty_axes_reduce_all(%data: tensor<?x?xf32>)
      -> tensor<1x1xf32> {
    %axes = arith.constant dense<> : tensor<0xi64>
    %output = "onnx.ReduceSum"(%data, %axes)
        {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<?x?xf32>, tensor<0xi64>) -> tensor<1x1xf32>
    return %output : tensor<1x1xf32>
  }

  // CHECK-LABEL: func.func @reduce_sum_empty_axes_reduce_all
  // CHECK: %[[EINIT:.*]] = tensor.empty() : tensor<1x1xf32>
  // CHECK: hip.reduce_sum({{.*}}) ins({{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<2xi64>) outs(%[[EINIT]] : tensor<1x1xf32>)

  // Empty constant axes preserve every dimension when noop_with_empty_axes=1.
  func.func @reduce_sum_empty_axes_noop(%data: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
    %axes = arith.constant dense<> : tensor<0xi64>
    %output = "onnx.ReduceSum"(%data, %axes)
        {keepdims = 0 : si64, noop_with_empty_axes = 1 : si64}
        : (tensor<?x?xf32>, tensor<0xi64>) -> tensor<?x?xf32>
    return %output : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @reduce_sum_empty_axes_noop
  // CHECK: %[[ED0:.*]] = tensor.dim %[[EDATA:.*]], %{{.*}} : tensor<?x?xf32>
  // CHECK: %[[ED1:.*]] = tensor.dim %[[EDATA]], %{{.*}} : tensor<?x?xf32>
  // CHECK: %[[ENINIT:.*]] = tensor.empty(%[[ED0]], %[[ED1]]) : tensor<?x?xf32>
  // CHECK: hip.reduce_sum({{.*}}) ins(%[[EDATA]], %{{.*}} : tensor<?x?xf32>, tensor<0xi64>) outs(%[[ENINIT]] : tensor<?x?xf32>) {keepdims = 0 : i64, noop_with_empty_axes = 1 : i64, normalized_axes = array<i64>}

  // An explicit ONNX NoValue is semantically the absent axes operand. The HIP
  // op still requires an axes tensor, so conversion must materialize all axes
  // rather than forwarding a `none` value.
  func.func @reduce_sum_none_axes(%data: tensor<?x?xf32>) -> tensor<f32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %output = "onnx.ReduceSum"(%data, %none)
        {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<?x?xf32>, none) -> tensor<f32>
    return %output : tensor<f32>
  }

  // CHECK-LABEL: func.func @reduce_sum_none_axes
  // CHECK: %[[AXES:.*]] = arith.constant dense<[0, 1]> : tensor<2xi64>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<f32>
  // CHECK: hip.reduce_sum({{.*}}) ins({{.*}}, %[[AXES]] : tensor<?x?xf32>, tensor<2xi64>) outs(%[[INIT]] : tensor<f32>) {keepdims = 0 : i64, normalized_axes = array<i64: 0, 1>}
  // CHECK-NOT: onnx.NoValue
}
