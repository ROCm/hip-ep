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

  // Rank-1 length-1 operands -- the topology emitted by every
  // *_mrope/range/Range node in Qwen3.5-9B/text.onnx. ONNX Range
  // accepts both rank-0 scalars and rank-1[1] tensors; the
  // conversion lowers both into the same hip.range body.
  //
  // When the output length is statically known (here:
  // ``tensor<4xi64>``) the per-operand tensor.extract for the
  // dynamic length computation is dead-code-eliminated, so the
  // hip.range ins receives the rank-1[1] tensor types directly.
  // The downstream hip-to-llvm lowering handles the extract.
  func.func @test_range_i64_rank1_constants() -> tensor<4xi64> {
    %s = arith.constant dense<0> : tensor<1xi64>
    %l = arith.constant dense<4> : tensor<1xi64>
    %d = arith.constant dense<1> : tensor<1xi64>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    return %r : tensor<4xi64>
  }
  // CHECK-LABEL: func.func @test_range_i64_rank1_constants
  // CHECK: hip.range
  // CHECK-SAME: tensor<1xi64>, tensor<1xi64>, tensor<1xi64>

  // Dynamic rank-1[1] operands -- runtime values, the exact
  // limit-of-Mul-of-Gather-of-Shape pattern from the mrope chains.
  // The dynamic output shape forces the host-side length
  // computation, which produces a tensor.extract on the rank-1[1]
  // operand (one index, %c0).
  func.func @test_range_i64_rank1_dynamic(%arg0: tensor<1xi64>, %arg1: tensor<1xi64>, %arg2: tensor<1xi64>) -> tensor<?xi64> {
    %r = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
    return %r : tensor<?xi64>
  }
  // CHECK-LABEL: func.func @test_range_i64_rank1_dynamic
  // CHECK: tensor.extract {{.*}} : tensor<1xi64>
  // CHECK: hip.range
  // CHECK-SAME: tensor<1xi64>, tensor<1xi64>, tensor<1xi64>

  // Mixed-rank operands: rank-0 start/delta + rank-1[1] limit.
  // Useful guard against an over-strict "all-or-nothing" rank
  // rule that would silently break the in-model topology mix
  // (start / delta are constant rank-1[1] initializers in Qwen
  // text.onnx; limit is a rank-1[1] Mul output; this test
  // exercises the per-operand rank check independently). The
  // emitted hip.range correctly preserves each operand's original
  // type in its ins list.
  func.func @test_range_i64_mixed_rank(%arg0: tensor<1xi64>) -> tensor<?xi64> {
    %s = arith.constant dense<0> : tensor<i64>
    %d = arith.constant dense<1> : tensor<i64>
    %r = "onnx.Range"(%s, %arg0, %d) : (tensor<i64>, tensor<1xi64>, tensor<i64>) -> tensor<?xi64>
    return %r : tensor<?xi64>
  }
  // CHECK-LABEL: func.func @test_range_i64_mixed_rank
  // CHECK: hip.range
  // CHECK-SAME: tensor<i64>, tensor<1xi64>, tensor<i64>
}
