// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify both ONNX ConstantOfShape lowering paths:
//
//   1. ConstantOfShapeFold (preferred): shape input is a compile-time
//      constant AND result type is fully static -> single splat
//      arith.constant; no runtime work.
//   2. ConstantOfShapeDynamic (fallback): shape input is non-constant OR
//      result has at least one dynamic dim -> tensor.splat whose dynamic
//      dim sizes are read out of the shape tensor at runtime via
//      tensor.extract + arith.index_cast.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: default value (fp32 zero) with shape from arith.constant.
  func.func @test_constant_of_shape_default() -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @test_constant_of_shape_default
    %shape = arith.constant dense<[2, 3]> : tensor<2xi64>
    %r = "onnx.ConstantOfShape"(%shape) : (tensor<2xi64>) -> tensor<2x3xf32>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK: arith.constant dense<0.000000e+00> : tensor<2x3xf32>

    return %r : tensor<2x3xf32>
  }

  // Test 2: custom fp32 value attribute.
  func.func @test_constant_of_shape_value_f32() -> tensor<2x2xf32> {
    // CHECK-LABEL: func.func @test_constant_of_shape_value_f32
    %shape = arith.constant dense<[2, 2]> : tensor<2xi64>
    %r = "onnx.ConstantOfShape"(%shape) {
      value = dense<1.500000e+00> : tensor<1xf32>
    } : (tensor<2xi64>) -> tensor<2x2xf32>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK: arith.constant dense<1.500000e+00> : tensor<2x2xf32>

    return %r : tensor<2x2xf32>
  }

  // Test 3: int64 value attribute.
  func.func @test_constant_of_shape_value_i64() -> tensor<3xi64> {
    // CHECK-LABEL: func.func @test_constant_of_shape_value_i64
    %shape = arith.constant dense<[3]> : tensor<1xi64>
    %r = "onnx.ConstantOfShape"(%shape) {
      value = dense<7> : tensor<1xi64>
    } : (tensor<1xi64>) -> tensor<3xi64>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK: arith.constant dense<7> : tensor<3xi64>

    return %r : tensor<3xi64>
  }

  // Note: the canonical "Shape -> ConstantOfShape" composition test
  // (transformer-style allocation of a zero KV/mask tensor) lives with
  // the Shape conversion in a separate PR. Once that PR lands, re-add a
  // composed test here.

  // Test 4: dynamic result shape with a non-constant shape input from a
  // func-arg -- the fold path bails (no compile-time shape attr AND
  // result is dynamic). The dynamic pattern now emits hip.constant_of_shape
  // with an `output_dim_specs` attribute encoding each dynamic dim as a
  // Category-B InputValueI64 leaf (no slot_ids).
  func.func @test_constant_of_shape_dynamic(%shape: tensor<2xi64>) -> tensor<?x?xf32> {
    // CHECK-LABEL: func.func @test_constant_of_shape_dynamic
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[SHAPE:.*]]: tensor<2xi64>)
    %r = "onnx.ConstantOfShape"(%shape) : (tensor<2xi64>) -> tensor<?x?xf32>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK-NOT: tensor.splat
    // CHECK: hip.constant_of_shape({{.*}})
    // CHECK-SAME: fill_value = 0
    // CHECK-SAME: output_data_type = 0
    // CHECK-SAME: output_dim_specs =
    // CHECK-NOT: slot_ids

    return %r : tensor<?x?xf32>
  }

  // Test 5: partially dynamic result with custom int value attribute --
  // the static dim becomes a Static DimSpec leaf and the dynamic dim
  // becomes an InputValueI64(input_index=1) leaf (Category B).
  func.func @test_constant_of_shape_partial_dynamic(%shape: tensor<2xi64>) -> tensor<3x?xi64> {
    // CHECK-LABEL: func.func @test_constant_of_shape_partial_dynamic
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[SHAPE:.*]]: tensor<2xi64>)
    %r = "onnx.ConstantOfShape"(%shape) {
      value = dense<42> : tensor<1xi64>
    } : (tensor<2xi64>) -> tensor<3x?xi64>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK-NOT: tensor.splat
    // CHECK: hip.constant_of_shape({{.*}})
    // CHECK-SAME: fill_value = 42
    // CHECK-SAME: output_data_type = 4
    // CHECK-SAME: output_dim_specs =
    // CHECK-NOT: slot_ids

    return %r : tensor<3x?xi64>
  }

  // Test 6 (Category C): the shape tensor is an intermediate value (here
  // we synthesise one via an onnx.Range whose `start` operand goes
  // through an onnx.Add so RangeConversion rejects Category B and falls
  // back to Category C). The dynamic pattern emits hip.constant_of_shape
  // with RuntimeSlot DimSpecs and a `slot_ids` attribute; slot IDs are
  // allocated per-module from `hipdnn.next_dyn_slot_id`.
  func.func @test_constant_of_shape_category_c(
      %a: tensor<i64>, %b: tensor<i64>, %c: tensor<i64>) -> tensor<?x?xi64> {
    // CHECK-LABEL: func.func @test_constant_of_shape_category_c
    %z = arith.constant dense<0> : tensor<i64>
    %a_intermediate = "onnx.Add"(%a, %z) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %shape = "onnx.Range"(%a_intermediate, %b, %c) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
    %r = "onnx.ConstantOfShape"(%shape) {
      value = dense<7> : tensor<1xi64>
    } : (tensor<?xi64>) -> tensor<?x?xi64>

    // CHECK: hip.constant_of_shape({{.*}})
    // CHECK-SAME: fill_value = 7
    // CHECK-SAME: slot_ids = array<i32:

    return %r : tensor<?x?xi64>
  }
}
