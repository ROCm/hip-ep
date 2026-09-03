// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the three ONNX ConstantOfShape lowering paths:
//
//   1. ConstantOfShapeAsScalar (highest benefit): result is consumed only by
//      ops that broadcast a scalar (today onnx.Where, onnx.Min) -> a rank-0 fill-value
//      buffer (tensor.empty + linalg.fill), elide the full-size buffer.
//   2. ConstantOfShapeFold: shape input is a compile-time constant AND result
//      type is fully static -> single splat arith.constant; no runtime work.
//   3. ConstantOfShapeDynamic (fallback): shape input is non-constant OR
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

  // Test 4: dynamic result shape with a non-constant shape input -- the
  // fold path bails (no compile-time shape attr AND result is dynamic)
  // and the dynamic pattern emits tensor.splat with per-dim
  // tensor.extract + index_cast for each dynamic result dim.
  func.func @test_constant_of_shape_dynamic(%shape: tensor<2xi64>) -> tensor<?x?xf32> {
    // CHECK-LABEL: func.func @test_constant_of_shape_dynamic
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[SHAPE:.*]]: tensor<2xi64>)
    %r = "onnx.ConstantOfShape"(%shape) : (tensor<2xi64>) -> tensor<?x?xf32>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[E0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]] : tensor<2xi64>
    // CHECK-DAG: %[[D0:.*]] = arith.index_cast %[[E0]] : i64 to index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[E1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<2xi64>
    // CHECK-DAG: %[[D1:.*]] = arith.index_cast %[[E1]] : i64 to index
    // CHECK: tensor.splat %[[ZERO]]{{\[}}%{{.*}}, %{{.*}}{{\]}} : tensor<?x?xf32>

    return %r : tensor<?x?xf32>
  }

  // Test 5: partially dynamic result with custom int value attribute --
  // the static dim is encoded in the result type and only the dynamic
  // dim gets a tensor.extract.
  func.func @test_constant_of_shape_partial_dynamic(%shape: tensor<2xi64>) -> tensor<3x?xi64> {
    // CHECK-LABEL: func.func @test_constant_of_shape_partial_dynamic
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[SHAPE:.*]]: tensor<2xi64>)
    %r = "onnx.ConstantOfShape"(%shape) {
      value = dense<42> : tensor<1xi64>
    } : (tensor<2xi64>) -> tensor<3x?xi64>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK-DAG: %[[VAL:.*]] = arith.constant 42 : i64
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[E1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<2xi64>
    // CHECK-DAG: %[[D1:.*]] = arith.index_cast %[[E1]] : i64 to index
    // CHECK: tensor.splat %[[VAL]]{{\[}}%{{.*}}{{\]}} : tensor<3x?xi64>

    return %r : tensor<3x?xi64>
  }

  // Test 6: ConstantOfShapeAsScalar -- the result is consumed only by
  // onnx.Where, so the fill-value buffer collapses to a rank-0
  // tensor.empty + linalg.fill instead of a full-size tensor.splat (which
  // would canonicalize to a host memref.global the GPU where-kernel cannot
  // read). The dynamic result dims are supplied by Where's other operands,
  // so no per-dim tensor.extract is emitted for the fill value.
  func.func @test_constant_of_shape_as_scalar_where(
      %mask: tensor<?x?xi1>, %ids: tensor<?x?xi64>, %shape: tensor<2xi64>)
      -> tensor<?x?xi64> {
    // CHECK-LABEL: func.func @test_constant_of_shape_as_scalar_where
    %c = "onnx.ConstantOfShape"(%shape) {
      value = dense<-100> : tensor<1xi64>
    } : (tensor<2xi64>) -> tensor<?x?xi64>
    %o = "onnx.Where"(%mask, %c, %ids)
      : (tensor<?x?xi1>, tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK-NOT: tensor.splat
    // CHECK-DAG: %[[V:.*]] = arith.constant -100 : i64
    // CHECK-DAG: %[[E:.*]] = tensor.empty() : tensor<i64>
    // CHECK: linalg.fill ins(%[[V]] : i64) outs(%[[E]] : tensor<i64>) -> tensor<i64>

    return %o : tensor<?x?xi64>
  }

  // Test 7: ConstantOfShapeAsScalar with onnx.Min -- same rank-0 fill path as
  // Where; Min broadcasts the scalar across the other operand's shape.
  func.func @test_constant_of_shape_as_scalar_min(
      %a: tensor<?x?xi64>, %shape: tensor<2xi64>) -> tensor<?x?xi64> {
    // CHECK-LABEL: func.func @test_constant_of_shape_as_scalar_min
    %c = "onnx.ConstantOfShape"(%shape) {
      value = dense<15> : tensor<1xi64>
    } : (tensor<2xi64>) -> tensor<?x?xi64>
    %o = "onnx.Min"(%a, %c) : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK-NOT: tensor.splat
    // CHECK-DAG: %[[V:.*]] = arith.constant 15 : i64
    // CHECK-DAG: %[[E:.*]] = tensor.empty() : tensor<i64>
    // CHECK: linalg.fill ins(%[[V]] : i64) outs(%[[E]] : tensor<i64>) -> tensor<i64>

    return %o : tensor<?x?xi64>
  }
}
