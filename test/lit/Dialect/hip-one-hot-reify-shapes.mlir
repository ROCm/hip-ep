// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s --check-prefix=RESOLVE
// RUN: hip-mlir-opt --hip-infer-shapes %s | FileCheck %s --check-prefix=INFER
// RUN: hip-mlir-opt --verify-each=0 --test-one-hot-reify-failure-atomic %s 2>&1 | FileCheck %s --check-prefix=ATOMIC

// A non-tensor.empty destination is the fallback authority for the runtime
// depth extent. Shape resolution inserts reified operations before hip.one_hot,
// so the fallback must read the dominating destination rather than %result.
// The other dimensions retain the exact indices-derived shape.
//
// RESOLVE-LABEL: func.func @non_empty_destination
// RESOLVE:         %[[INDEX_DIM:.*]] = tensor.dim %[[INDICES:.*]], %{{.*}} : tensor<?x4xi64>
// RESOLVE:         %[[DEPTH_DIM:.*]] = tensor.dim %[[OUTPUT:.*]], %{{.*}} : tensor<?x?x?xf32>
// RESOLVE-NOT:     hip.one_hot
// RESOLVE:         return %[[INDEX_DIM]], %[[DEPTH_DIM]], %{{.*}} : index, index, index
//
// INFER-LABEL: func.func @non_empty_destination
// INFER:         tensor.dim %{{.*}} : tensor<?x4xi64>
// INFER:         tensor.dim %{{.*}} : tensor<?x?x?xf32>
// INFER:         hip.one_hot
func.func @non_empty_destination(
    %ctx: !hip.context, %indices: tensor<?x4xi64>, %depth: tensor<i64>,
    %values: tensor<2xf32>, %output: tensor<?x?x?xf32>)
    -> (index, index, index) {
  %result = hip.one_hot(%ctx)
      ins(%indices, %depth, %values :
          tensor<?x4xi64>, tensor<i64>, tensor<2xf32>)
      outs(%output : tensor<?x?x?xf32>)
      {axis = 1 : i64} : tensor<?x?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %result, %c2 : tensor<?x?x?xf32>
  return %d0, %d1, %d2 : index, index, index
}

// -----

// A constant depth payload is a stronger semantic source than outs. All
// dimensions are exact, so reification emits no tensor.dim operations.
//
// RESOLVE-LABEL: func.func @constant_depth_exact_shape
// RESOLVE-NOT:     tensor.dim
// RESOLVE:         return %{{.*}}, %{{.*}}, %{{.*}} : index, index, index
func.func @constant_depth_exact_shape(
    %ctx: !hip.context, %indices: tensor<2x4xi32>,
    %values: tensor<2xf32>, %output: tensor<2x7x4xf32>)
    -> (index, index, index) {
  %depth = arith.constant dense<7> : tensor<i64>
  %result = hip.one_hot(%ctx)
      ins(%indices, %depth, %values :
          tensor<2x4xi32>, tensor<i64>, tensor<2xf32>)
      outs(%output : tensor<2x7x4xf32>)
      {axis = 1 : i64} : tensor<2x7x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %result, %c0 : tensor<2x7x4xf32>
  %d1 = tensor.dim %result, %c1 : tensor<2x7x4xf32>
  %d2 = tensor.dim %result, %c2 : tensor<2x7x4xf32>
  return %d0, %d1, %d2 : index, index, index
}

// -----

// Dense hip.constant carriers preserve the same exact-depth behavior as
// arith.constant tensors.
//
// RESOLVE-LABEL: func.func @hip_constant_depth_exact_shape
// RESOLVE-NOT:     tensor.dim
// RESOLVE:         return %{{.*}}, %{{.*}}, %{{.*}} : index, index, index
func.func @hip_constant_depth_exact_shape(
    %ctx: !hip.context, %indices: tensor<2x4xi32>,
    %values: tensor<2xf32>, %output: tensor<2x7x4xf32>)
    -> (index, index, index) {
  %depth = hip.constant {value = dense<7> : tensor<i64>} : tensor<i64>
  %result = hip.one_hot(%ctx)
      ins(%indices, %depth, %values :
          tensor<2x4xi32>, tensor<i64>, tensor<2xf32>)
      outs(%output : tensor<2x7x4xf32>)
      {axis = 1 : i64} : tensor<2x7x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %result, %c0 : tensor<2x7x4xf32>
  %d1 = tensor.dim %result, %c1 : tensor<2x7x4xf32>
  %d2 = tensor.dim %result, %c2 : tensor<2x7x4xf32>
  return %d0, %d1, %d2 : index, index, index
}

// -----

// The tool-only probe changes the valid destination's depth-axis type from 7
// to 8 immediately before calling reification. The dense hip.constant remains
// the semantic depth authority, so reification must reject the contradiction
// without changing the block, attributes, types, or shape list. Expensive
// pattern API checks enforce the same mutation contract for the direct
// resolve-shaped-type-result-dims run above.
//
// ATOMIC: remark: failed OneHot reification left IR unchanged
// ATOMIC-NOT: failed OneHot reification mutated IR
func.func @failure_is_atomic(
    %ctx: !hip.context, %indices: tensor<2x4xi64>,
    %values: tensor<2xf32>) {
  %depth = hip.constant {value = dense<7> : tensor<i64>} : tensor<i64>
  %output = tensor.empty() : tensor<2x7x4xf32>
  %result = hip.one_hot(%ctx)
      ins(%indices, %depth, %values :
          tensor<2x4xi64>, tensor<i64>, tensor<2xf32>)
      outs(%output : tensor<2x7x4xf32>)
      {axis = 1 : i64, test.one_hot_reify_failure_atomic}
      : tensor<2x7x4xf32>
  return
}
