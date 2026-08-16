// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @constant_positive
  // CHECK-NOT: hip.readback_control
  // CHECK-NOT: hip.slice
  // CHECK: tensor.extract_slice
  func.func @constant_positive(%input: tensor<4x6xf32>) -> tensor<2x6xf32> {
    %starts = arith.constant dense<[1]> : tensor<1xi64>
    %ends = arith.constant dense<[3]> : tensor<1xi64>
    %axes = arith.constant dense<[0]> : tensor<1xi64>
    %steps = arith.constant dense<[1]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<4x6xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<2x6xf32>
    return %r : tensor<2x6xf32>
  }

  // Omitted axes cover only the parameter count, not the whole data rank.
  // CHECK-LABEL: func.func @omitted_axes_shorter_than_rank
  // CHECK: tensor.extract_slice
  func.func @omitted_axes_shorter_than_rank(
      %input: tensor<4x6xf32>) -> tensor<2x6xf32> {
    %starts = arith.constant dense<[1]> : tensor<1xi64>
    %ends = arith.constant dense<[3]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends)
        : (tensor<4x6xf32>, tensor<1xi64>, tensor<1xi64>)
          -> tensor<2x6xf32>
    return %r : tensor<2x6xf32>
  }

  // CHECK-LABEL: func.func @constant_negative
  // CHECK-NOT: hip.readback_control
  // CHECK: tensor.empty
  // CHECK: hip.slice
  // CHECK-SAME: valid(
  // CHECK-SAME: starts(
  // CHECK-SAME: steps(
  // CHECK-SAME: extents(
  func.func @constant_negative(%input: tensor<6xf32>) -> tensor<3xf32> {
    %starts = arith.constant dense<[5]> : tensor<1xi64>
    %ends = arith.constant dense<[0]> : tensor<1xi64>
    %axes = arith.constant dense<[0]> : tensor<1xi64>
    %steps = arith.constant dense<[-2]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<6xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
    return %r : tensor<3xf32>
  }

  // Runtime i32 is sign-extended by one generic grouped readback. Constant
  // ends is consumed directly and is not included as a readback source.
  // CHECK-LABEL: func.func @runtime_i32_mixed_constant
  // CHECK-COUNT-1: hip.readback_control
  // CHECK-SAME: tensor<1xi32>
  // CHECK: hip.slice
  func.func @runtime_i32_mixed_constant(
      %input: tensor<8xf32>, %starts: tensor<1xi32>) -> tensor<?xf32> {
    %ends = arith.constant dense<[7]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends)
        : (tensor<8xf32>, tensor<1xi32>, tensor<1xi64>) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }

  // All genuinely runtime controls share one readback, in operand-major order.
  // CHECK-LABEL: func.func @runtime_mixed_i32_i64
  // CHECK-COUNT-1: hip.readback_control
  // CHECK-SAME: tensor<2xi32>, tensor<2xi64>, tensor<2xi32>, tensor<2xi64>
  // CHECK: hip.slice
  func.func @runtime_mixed_i32_i64(
      %input: tensor<4x6xf32>, %starts: tensor<2xi32>,
      %ends: tensor<2xi64>, %axes: tensor<2xi32>,
      %steps: tensor<2xi64>) -> tensor<?x?xf32> {
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<4x6xf32>, tensor<2xi32>, tensor<2xi64>,
           tensor<2xi32>, tensor<2xi64>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @constant_empty
  // CHECK-NOT: hip.readback_control
  // CHECK: hip.slice
  // CHECK: return
  func.func @constant_empty(%input: tensor<1x8x4xf32>)
      -> tensor<1x0x4xf32> {
    %starts = arith.constant dense<[0]> : tensor<1xi64>
    %ends = arith.constant dense<[0]> : tensor<1xi64>
    %axes = arith.constant dense<[1]> : tensor<1xi64>
    %steps = arith.constant dense<[-1]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<1x8x4xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<1x0x4xf32>
    return %r : tensor<1x0x4xf32>
  }

  // CHECK-LABEL: func.func @negative_int64_min_step
  // CHECK-NOT: hip.readback_control
  // CHECK: hip.slice
  func.func @negative_int64_min_step(%input: tensor<10xf32>)
      -> tensor<1xf32> {
    %starts = arith.constant dense<[9]> : tensor<1xi64>
    %ends = arith.constant dense<[-9223372036854775808]> : tensor<1xi64>
    %axes = arith.constant dense<[0]> : tensor<1xi64>
    %steps = arith.constant dense<[-9223372036854775808]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
    return %r : tensor<1xf32>
  }
}
