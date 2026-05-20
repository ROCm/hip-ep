// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the two onnx.Size paths in convert-onnx-to-hip:
//
//  * Static-shaped inputs are constant-folded into a rank-0 int64
//    `arith.constant`. No HIP op, no runtime symbol -- the historical
//    fast path.
//  * Dynamic-shaped inputs (at least one `?` dim) get rewritten into a
//    `hip.size` DPS op with a `tensor.empty()` rank-0 init buffer; the
//    HipToLLVM pass later computes `prod(input.shape)` at runtime.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: 3D static input -> 2*3*4 = 24 elements.
  func.func @test_size_3d(%input: tensor<2x3x4xf32>) -> tensor<i64> {
    // CHECK-LABEL: func.func @test_size_3d
    %r = "onnx.Size"(%input) : (tensor<2x3x4xf32>) -> tensor<i64>
    // CHECK-NOT: onnx.Size
    // CHECK: arith.constant dense<24> : tensor<i64>
    return %r : tensor<i64>
  }

  // Test 2: rank-1 input — element count equals the single dim.
  func.func @test_size_1d(%input: tensor<128xf16>) -> tensor<i64> {
    // CHECK-LABEL: func.func @test_size_1d
    %r = "onnx.Size"(%input) : (tensor<128xf16>) -> tensor<i64>
    // CHECK-NOT: onnx.Size
    // CHECK: arith.constant dense<128> : tensor<i64>
    return %r : tensor<i64>
  }

  // Test 3: scalar (rank-0) input — 1 element.
  func.func @test_size_scalar(%input: tensor<f32>) -> tensor<i64> {
    // CHECK-LABEL: func.func @test_size_scalar
    %r = "onnx.Size"(%input) : (tensor<f32>) -> tensor<i64>
    // CHECK-NOT: onnx.Size
    // CHECK: arith.constant dense<1> : tensor<i64>
    return %r : tensor<i64>
  }

  // Test 4: integer input — dtype is irrelevant, only the shape matters.
  func.func @test_size_i64(%input: tensor<5x7xi64>) -> tensor<i64> {
    // CHECK-LABEL: func.func @test_size_i64
    %r = "onnx.Size"(%input) : (tensor<5x7xi64>) -> tensor<i64>
    // CHECK-NOT: onnx.Size
    // CHECK: arith.constant dense<35> : tensor<i64>
    return %r : tensor<i64>
  }

  // Test 5: fully dynamic 2D input — must emit hip.size, not arith.constant.
  func.func @test_size_dyn_2d(%input: tensor<?x?xf32>) -> tensor<i64> {
    // CHECK-LABEL: func.func @test_size_dyn_2d
    // CHECK-SAME: (%[[CTX5:.*]]: !hip.context, %[[IN5:.*]]: tensor<?x?xf32>)
    %r = "onnx.Size"(%input) : (tensor<?x?xf32>) -> tensor<i64>
    // CHECK-NOT: onnx.Size
    // CHECK-NOT: arith.constant dense
    // CHECK: %[[INIT5:.*]] = tensor.empty() : tensor<i64>
    // CHECK: hip.size(%[[CTX5]]) ins(%[[IN5]] : tensor<?x?xf32>) outs(%[[INIT5]] : tensor<i64>)
    return %r : tensor<i64>
  }

  // Test 6: partially dynamic (mix of static + dynamic dims). Same outcome.
  func.func @test_size_dyn_mixed(%input: tensor<2x?x4xf16>) -> tensor<i64> {
    // CHECK-LABEL: func.func @test_size_dyn_mixed
    // CHECK-SAME: (%[[CTX6:.*]]: !hip.context, %[[IN6:.*]]: tensor<2x?x4xf16>)
    %r = "onnx.Size"(%input) : (tensor<2x?x4xf16>) -> tensor<i64>
    // CHECK-NOT: onnx.Size
    // CHECK-NOT: arith.constant dense
    // CHECK: %[[INIT6:.*]] = tensor.empty() : tensor<i64>
    // CHECK: hip.size(%[[CTX6]]) ins(%[[IN6]] : tensor<2x?x4xf16>) outs(%[[INIT6]] : tensor<i64>)
    return %r : tensor<i64>
  }
}
