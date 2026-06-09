// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata in convert-onnx-to-hip.
  func.func @main_graph(%arg0: tensor<i32>) -> tensor<i32> {
    return %arg0 : tensor<i32>
  }

  // Scalar constant operands still lower through hip.range (no compile-time fold).
  func.func @test_range_i32_constants() -> tensor<4xi32> {
    %s = arith.constant dense<2> : tensor<i32>
    %l = arith.constant dense<10> : tensor<i32>
    %d = arith.constant dense<2> : tensor<i32>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
  // CHECK-LABEL: func.func @test_range_i32_constants
  // CHECK-SAME: !hip.context
  // CHECK: tensor.empty
  // CHECK: hip.range

  // Dynamic operands lower to hip.range with DPS init tensor.
  func.func @test_range_i32_dynamic(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?xi32> {
    %r = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
    return %r : tensor<?xi32>
  }
  // CHECK-LABEL: func.func @test_range_i32_dynamic
  // CHECK: tensor.empty
  // CHECK: hip.range

  func.func @test_range_i16() -> tensor<4xi16> {
    %s = arith.constant dense<0> : tensor<i16>
    %l = arith.constant dense<4> : tensor<i16>
    %d = arith.constant dense<1> : tensor<i16>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<i16>, tensor<i16>, tensor<i16>) -> tensor<4xi16>
    return %r : tensor<4xi16>
  }
  // CHECK-LABEL: func.func @test_range_i16
  // CHECK: hip.range

  func.func @test_range_i64() -> tensor<4xi64> {
    %s = arith.constant dense<0> : tensor<i64>
    %l = arith.constant dense<4> : tensor<i64>
    %d = arith.constant dense<1> : tensor<i64>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4xi64>
    return %r : tensor<4xi64>
  }
  // CHECK-LABEL: func.func @test_range_i64
  // CHECK: hip.range

  func.func @test_range_f64() -> tensor<4xf64> {
    %s = arith.constant dense<0.0> : tensor<f64>
    %l = arith.constant dense<4.0> : tensor<f64>
    %d = arith.constant dense<1.0> : tensor<f64>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<4xf64>
    return %r : tensor<4xf64>
  }
  // CHECK-LABEL: func.func @test_range_f64
  // CHECK: hip.range

  // Empty ranges are still lowered to hip.range. Runtime gets an empty output.
  func.func @test_range_empty_pos_i32() -> tensor<0xi32> {
    %s = arith.constant dense<5> : tensor<i32>
    %l = arith.constant dense<2> : tensor<i32>
    %d = arith.constant dense<1> : tensor<i32>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<0xi32>
    return %r : tensor<0xi32>
  }
  // CHECK-LABEL: func.func @test_range_empty_pos_i32
  // CHECK: tensor.empty
  // CHECK: hip.range

  // Negative delta with increasing interval also yields empty output.
  func.func @test_range_empty_neg_f32() -> tensor<0xf32> {
    %s = arith.constant dense<0.0> : tensor<f32>
    %l = arith.constant dense<5.0> : tensor<f32>
    %d = arith.constant dense<-1.0> : tensor<f32>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<0xf32>
    return %r : tensor<0xf32>
  }
  // CHECK-LABEL: func.func @test_range_empty_neg_f32
  // CHECK: tensor.empty
  // CHECK: hip.range

  // ONNX Range scalars may be rank-1 size-1 (tensor<1xi64>), not just rank-0.
  // Such operands must still match and lower to hip.range.
  func.func @test_range_rank1_dynamic(%s: tensor<1xi64>, %l: tensor<1xi64>, %d: tensor<1xi64>) -> tensor<?xi64> {
    %r = "onnx.Range"(%s, %l, %d) : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
    return %r : tensor<?xi64>
  }
  // CHECK-LABEL: func.func @test_range_rank1_dynamic
  // CHECK: hip.range

  // from_elements scalar producers are traced directly to their element, so
  // the count formula stays free of tensor.extract (keeps the dynamic-size
  // alloc hoistable by hip-pool-allocs). All three operands use runtime
  // index args so the from_elements are not constant-folded to tensor
  // constants (which would fall back to the tensor.extract path).
  func.func @test_range_from_elements(%s: index, %l: index, %d: index) -> tensor<?xi64> {
    %sc = arith.index_cast %s : index to i64
    %lc = arith.index_cast %l : index to i64
    %dc = arith.index_cast %d : index to i64
    %start = tensor.from_elements %sc : tensor<1xi64>
    %limit = tensor.from_elements %lc : tensor<1xi64>
    %delta = tensor.from_elements %dc : tensor<1xi64>
    %r = "onnx.Range"(%start, %limit, %delta) : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
    return %r : tensor<?xi64>
  }
  // CHECK-LABEL: func.func @test_range_from_elements
  // CHECK-NOT: tensor.extract
  // CHECK: hip.range
}
