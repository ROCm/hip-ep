// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the three ONNX ConstantOfShape lowering paths:
//
//   1. ConstantOfShapeAsScalar (highest benefit): result is consumed only by
//      ops that broadcast a scalar (today onnx.Where) -> a rank-0 fill-value
//      buffer (tensor.empty + linalg.fill), elide the full-size buffer.
//   2. ConstantOfShapeFold: shape input is a compile-time constant AND result
//      type is fully static -> single splat arith.constant; no runtime work.
//   3. ConstantOfShapeDynamic (fallback): shape input is non-constant OR
//      result has at least one dynamic dim -> tensor.splat. Compile-time
//      payloads remain constants; device payloads use one status-bearing,
//      non-negative grouped readback.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: default value with shape from an imported ONNX constant. The first
  // carrier sweep makes the dense payload visible to ordinary compute
  // conversion; no ConstantOfShape-specific pre-carrier invocation is needed.
  func.func @test_constant_of_shape_default() -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @test_constant_of_shape_default
    %shape = "onnx.Constant"() {
      value = dense<[2, 3]> : tensor<2xi64>
    } : () -> tensor<2xi64>
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
  // and the dynamic pattern emits one grouped synchronized readback. The
  // status guards every extent before its checked i64-to-index conversion.
  func.func @test_constant_of_shape_dynamic(%shape: tensor<2xi64>) -> tensor<?x?xf32> {
    // CHECK-LABEL: func.func @test_constant_of_shape_dynamic
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[SHAPE:.*]]: tensor<2xi64>)
    %r = "onnx.ConstantOfShape"(%shape) : (tensor<2xi64>) -> tensor<?x?xf32>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK-DAG: %[[Z64:.*]] = arith.constant 0 : i64
    // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK-NOT: tensor.extract
    // CHECK-COUNT-1: %[[VALID:.*]], %[[VALUES:.*]]:2 = hip.readback_control(%[[CTX]], %[[SHAPE]] : tensor<2xi64>) {require_non_negative = true} -> (i1, i64, i64)
    // CHECK: %[[S0:.*]] = arith.select %[[VALID]], %[[VALUES]]#0, %[[Z64]] : i64
    // CHECK: %[[D0:.*]] = arith.index_cast %[[S0]] : i64 to index
    // CHECK: %[[S1:.*]] = arith.select %[[VALID]], %[[VALUES]]#1, %[[Z64]] : i64
    // CHECK: %[[D1:.*]] = arith.index_cast %[[S1]] : i64 to index
    // CHECK: tensor.splat %[[ZERO]]{{\[}}%[[D0]], %[[D1]]{{\]}} : tensor<?x?xf32>
    // CHECK-NOT: tensor.extract

    return %r : tensor<?x?xf32>
  }

  // Test 5: a constant payload with a partially dynamic result stays entirely
  // on the compile-time path. The static result dim is cross-checked and the
  // dynamic dim is materialized directly as an index constant.
  func.func @test_constant_of_shape_partial_dynamic() -> tensor<3x?xi64> {
    // CHECK-LABEL: func.func @test_constant_of_shape_partial_dynamic
    %shape = arith.constant dense<[3, 5]> : tensor<2xi64>
    %r = "onnx.ConstantOfShape"(%shape) {
      value = dense<42> : tensor<1xi64>
    } : (tensor<2xi64>) -> tensor<3x?xi64>

    // CHECK-NOT: onnx.ConstantOfShape
    // CHECK-NOT: hip.readback
    // CHECK-NOT: tensor.extract
    // CHECK-DAG: %[[VAL:.*]] = arith.constant 42 : i64
    // CHECK-DAG: %[[D1:.*]] = arith.constant 5 : index
    // CHECK: tensor.splat %[[VAL]]{{\[}}%[[D1]]{{\]}} : tensor<3x?xi64>

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

  // Test 7: i32 device payloads are sign-extended by the same grouped
  // readback contract and then converted from its i64 results to index.
  func.func @test_constant_of_shape_dynamic_i32(
      %shape: tensor<1xi32>) -> tensor<?xf32> {
    // CHECK-LABEL: func.func @test_constant_of_shape_dynamic_i32
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[SHAPE:.*]]: tensor<1xi32>)
    %r = "onnx.ConstantOfShape"(%shape)
      : (tensor<1xi32>) -> tensor<?xf32>

    // CHECK-NOT: tensor.extract
    // CHECK-COUNT-1: %[[VALID:.*]], %[[VALUE:.*]] = hip.readback_control(%[[CTX]], %[[SHAPE]] : tensor<1xi32>) {require_non_negative = true} -> (i1, i64)
    // CHECK: arith.select %[[VALID]], %[[VALUE]], %{{.*}} : i64
    // CHECK: arith.index_cast %{{.*}} : i64 to index
    // CHECK: tensor.splat

    return %r : tensor<?xf32>
  }
}
