// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Slice is correctly lowered to hip.slice operation
// in tensor-first mode.
//
// This test validates:
// - Slice operation lowering (onnx.Slice -> hip.slice)
// - 3-operand form: data, starts, ends (axes/steps defaulted)
// - 4-operand form: data, starts, ends, axes
// - 5-operand form: data, starts, ends, axes, steps
// - onnx.NoValue handling for absent optional operands
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // ===== Test 1: 3-operand Slice (data, starts, ends only) =====
  func.func @main_graph(%data: tensor<4x8xf32>, %starts: tensor<2xi64>, %ends: tensor<2xi64>) -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<4x8xf32>, %[[STARTS:.*]]: tensor<2xi64>, %[[ENDS:.*]]: tensor<2xi64>) -> tensor<2x4xf32>

    %output = "onnx.Slice"(%data, %starts, %ends) : (tensor<4x8xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x4xf32>

    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x4xf32>
    // CHECK: hip.slice(%[[CTX]]) ins(%[[DATA]], %[[STARTS]], %[[ENDS]] : tensor<4x8xf32>, tensor<2xi64>, tensor<2xi64>) outs(%[[INIT]] : tensor<2x4xf32>)
    // CHECK-NOT: onnx.Slice
    // CHECK-NOT: hip.alloc

    return %output : tensor<2x4xf32>
  }

  // ===== Test 2: 4-operand Slice (with axes) =====
  func.func @test_slice_with_axes(%data: tensor<4x8xf32>, %starts: tensor<1xi64>, %ends: tensor<1xi64>, %axes: tensor<1xi64>) -> tensor<4x4xf32> {
    // CHECK-LABEL: func.func @test_slice_with_axes
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<4x8xf32>, %[[STARTS:.*]]: tensor<1xi64>, %[[ENDS:.*]]: tensor<1xi64>, %[[AXES:.*]]: tensor<1xi64>) -> tensor<4x4xf32>

    %output = "onnx.Slice"(%data, %starts, %ends, %axes) : (tensor<4x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4x4xf32>

    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x4xf32>
    // CHECK: hip.slice(%[[CTX]]) ins(%[[DATA]], %[[STARTS]], %[[ENDS]], %[[AXES]] : tensor<4x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) outs(%[[INIT]] : tensor<4x4xf32>)
    // CHECK-NOT: onnx.Slice

    return %output : tensor<4x4xf32>
  }

  // ===== Test 3: 5-operand Slice (with axes and steps) =====
  func.func @test_slice_with_axes_and_steps(
      %data: tensor<4x8xf32>, %starts: tensor<2xi64>, %ends: tensor<2xi64>,
      %axes: tensor<2xi64>, %steps: tensor<2xi64>) -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @test_slice_with_axes_and_steps
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<4x8xf32>, %[[STARTS:.*]]: tensor<2xi64>, %[[ENDS:.*]]: tensor<2xi64>, %[[AXES:.*]]: tensor<2xi64>, %[[STEPS:.*]]: tensor<2xi64>) -> tensor<2x4xf32>

    %output = "onnx.Slice"(%data, %starts, %ends, %axes, %steps) : (tensor<4x8xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x4xf32>

    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x4xf32>
    // CHECK: hip.slice(%[[CTX]]) ins(%[[DATA]], %[[STARTS]], %[[ENDS]], %[[AXES]], %[[STEPS]] : tensor<4x8xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) outs(%[[INIT]] : tensor<2x4xf32>)
    // CHECK-NOT: onnx.Slice

    return %output : tensor<2x4xf32>
  }

  // ===== Test 4: onnx.NoValue for axes (steps provided) =====
  func.func @test_slice_no_axes_with_steps(
      %data: tensor<4x8xf32>, %starts: tensor<2xi64>, %ends: tensor<2xi64>,
      %steps: tensor<2xi64>) -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @test_slice_no_axes_with_steps
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<4x8xf32>, %[[STARTS:.*]]: tensor<2xi64>, %[[ENDS:.*]]: tensor<2xi64>, %[[STEPS:.*]]: tensor<2xi64>) -> tensor<2x4xf32>

    %none = "onnx.NoValue"() {value} : () -> none
    %output = "onnx.Slice"(%data, %starts, %ends, %none, %steps) : (tensor<4x8xf32>, tensor<2xi64>, tensor<2xi64>, none, tensor<2xi64>) -> tensor<2x4xf32>

    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x4xf32>
    // CHECK: hip.slice(%[[CTX]]) ins(%[[DATA]], %[[STARTS]], %[[ENDS]], %[[STEPS]] : tensor<4x8xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) outs(%[[INIT]] : tensor<2x4xf32>)
    // CHECK-NOT: onnx.Slice
    // CHECK-NOT: onnx.NoValue

    return %output : tensor<2x4xf32>
  }
}
