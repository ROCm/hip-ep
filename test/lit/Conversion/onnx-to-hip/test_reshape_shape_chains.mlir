// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip \
// RUN:   --canonicalize --cse --hip-resolve-tensor-dims -o %t
// RUN: FileCheck %s --check-prefix=CHECK < %t
// RUN: FileCheck %s --check-prefix=SCALARS < %t

func.func @main_graph(%data: tensor<1xf16>) -> tensor<1xf16> {
  return %data : tensor<1xf16>
}

// Keep the order selected by Gather. Do not replace it with input axis order.
// CHECK-LABEL: func.func @permuted_shape_chain(
// CHECK-SAME: %[[DATA:[a-zA-Z0-9_]+]]: tensor<?x?x4xf16>
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[DATA]], %[[C0]]
// CHECK: %[[S0:.*]] = arith.index_cast %[[D0]] : index to i64
// CHECK: %[[D1:.*]] = tensor.dim %[[DATA]], %[[C1]]
// CHECK: %[[S1:.*]] = arith.index_cast %[[D1]] : index to i64
// CHECK: %[[SHAPE:.*]] = tensor.from_elements %[[S1]], %[[S0]], %[[TWO]], %[[TWO]] : tensor<4xi64>
// CHECK: %[[RESULT:.*]] = tensor.reshape %[[DATA]](%[[SHAPE]])
// CHECK: return %[[RESULT]]
// SCALARS-LABEL: func.func @permuted_shape_chain(
// SCALARS-NOT: hip.readback_scalar
// SCALARS-NOT: arith.divsi
// SCALARS: return
func.func @permuted_shape_chain(%data: tensor<?x?x4xf16>) -> tensor<?x?x?x?xf16> {
  %idx0 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %idx1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %two = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %shape = "onnx.Shape"(%data) : (tensor<?x?x4xf16>) -> tensor<3xi64>
  %d0 = "onnx.Gather"(%shape, %idx0) {axis = 0 : si64} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
  %d1 = "onnx.Gather"(%shape, %idx1) {axis = 0 : si64} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
  %u0 = "onnx.Unsqueeze"(%d0, %axes) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %u1 = "onnx.Unsqueeze"(%d1, %axes) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %target = "onnx.Concat"(%u1, %u0, %two, %two) {axis = 0 : si64}
    : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %result = "onnx.Reshape"(%data, %target) {allowzero = 1 : si64}
    : (tensor<?x?x4xf16>, tensor<4xi64>) -> tensor<?x?x?x?xf16>
  return %result : tensor<?x?x?x?xf16>
}

// The second shape uses the first Reshape result. Resolve both to input extents.
// CHECK-LABEL: func.func @dependent_reshapes(
// CHECK-SAME: %[[DATA:[a-zA-Z0-9_]+]]: tensor<?x?xf16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[DATA]], %[[C0]]
// CHECK: %[[D1:.*]] = tensor.dim %[[DATA]], %[[C1]]
// CHECK: %[[RESULT:.*]] = tensor.expand_shape %[[DATA]] {{.*}} output_shape [%[[D0]], 1, %[[D1]], 1]
// CHECK: return %[[RESULT]]
// SCALARS-LABEL: func.func @dependent_reshapes(
// SCALARS-NOT: hip.readback_scalar
// SCALARS-NOT: arith.divsi
// SCALARS: return
func.func @dependent_reshapes(%data: tensor<?x?xf16>) -> tensor<?x1x?x1xf16> {
  %idx0 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %idx1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %one = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %shape = "onnx.Shape"(%data) : (tensor<?x?xf16>) -> tensor<2xi64>
  %d0 = "onnx.Gather"(%shape, %idx0) {axis = 0 : si64} : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
  %d1 = "onnx.Gather"(%shape, %idx1) {axis = 0 : si64} : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
  %u0 = "onnx.Unsqueeze"(%d0, %axes) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %u1 = "onnx.Unsqueeze"(%d1, %axes) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %target0 = "onnx.Concat"(%u0, %one, %u1) {axis = 0 : si64}
    : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %first = "onnx.Reshape"(%data, %target0) {allowzero = 1 : si64}
    : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x1x?xf16>
  %first_shape = "onnx.Shape"(%first) : (tensor<?x1x?xf16>) -> tensor<3xi64>
  %target1 = "onnx.Concat"(%first_shape, %one) {axis = 0 : si64}
    : (tensor<3xi64>, tensor<1xi64>) -> tensor<4xi64>
  %second = "onnx.Reshape"(%first, %target1) {allowzero = 1 : si64}
    : (tensor<?x1x?xf16>, tensor<4xi64>) -> tensor<?x1x?x1xf16>
  return %second : tensor<?x1x?x1xf16>
}

// Both Reshapes use the selected shape. Keep the full vector for its other user.
// CHECK-LABEL: func.func @shared_shape_slice(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: tensor<?x?xf16>, %[[RHS:[a-zA-Z0-9_]+]]: tensor<?x?xf16>
// CHECK-DAG: %[[SEVEN:.*]] = arith.constant 7 : i64
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[LHS]], %[[C0]]
// CHECK: %[[SIZE:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK: %[[FULL:.*]] = tensor.from_elements %[[SIZE]], %[[ONE]], %[[SEVEN]] : tensor<3xi64>
// CHECK: %[[TARGET:.*]] = tensor.from_elements %[[SIZE]], %[[ONE]] : tensor<2xi64>
// CHECK: %[[LEFT:.*]] = tensor.reshape %[[LHS]](%[[TARGET]])
// CHECK: %[[RIGHT:.*]] = tensor.reshape %[[RHS]](%[[TARGET]])
// CHECK: return %[[LEFT]], %[[RIGHT]], %[[FULL]]
// SCALARS-LABEL: func.func @shared_shape_slice(
// SCALARS-NOT: hip.readback_scalar
// SCALARS-NOT: arith.divsi
// SCALARS: return
func.func @shared_shape_slice(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>)
    -> (tensor<?x1xf16>, tensor<?x1xf16>, tensor<3xi64>) {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %lhs, %c0 : tensor<?x?xf16>
  %size = arith.index_cast %dim : index to i64
  %one = arith.constant 1 : i64
  %seven = arith.constant 7 : i64
  %full = tensor.from_elements %size, %one, %seven : tensor<3xi64>
  %start = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %end = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %axis = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %slice = "onnx.Slice"(%full, %start, %end, %axis)
    : (tensor<3xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %left = "onnx.Reshape"(%lhs, %slice) {allowzero = 1 : si64}
    : (tensor<?x?xf16>, tensor<2xi64>) -> tensor<?x1xf16>
  %right = "onnx.Reshape"(%rhs, %slice) {allowzero = 1 : si64}
    : (tensor<?x?xf16>, tensor<2xi64>) -> tensor<?x1xf16>
  return %left, %right, %full : tensor<?x1xf16>, tensor<?x1xf16>, tensor<3xi64>
}

// A branch can select unknown shape data. Read the selected tensor, not one input.
// CHECK-LABEL: func.func @unknown_shape_join(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !hip.context, %[[DATA:[a-zA-Z0-9_]+]]: tensor<?x?xf16>, %[[UNKNOWN:[a-zA-Z0-9_]+]]: tensor<3xi64>, %[[COND:[a-zA-Z0-9_]+]]: i1
// CHECK: %[[JOINED:.*]] = arith.select %[[COND]], %{{[a-zA-Z0-9_]+}}, %[[UNKNOWN]] : tensor<3xi64>
// CHECK: %[[SLICE0:.*]] = tensor.extract_slice %[[JOINED]][0] [1] [1]
// CHECK: %[[SCALAR0:.*]] = tensor.collapse_shape %[[SLICE0]] []
// CHECK: %[[SIZE0:.*]] = hip.readback_scalar(%[[CTX]], %[[SCALAR0]] : tensor<i64>) -> i64
// CHECK: %[[SLICE1:.*]] = tensor.extract_slice %[[JOINED]][1] [1] [1]
// CHECK: %[[SCALAR1:.*]] = tensor.collapse_shape %[[SLICE1]] []
// CHECK: %[[SIZE1:.*]] = hip.readback_scalar(%[[CTX]], %[[SCALAR1]] : tensor<i64>) -> i64
// CHECK: %[[SLICE2:.*]] = tensor.extract_slice %[[JOINED]][2] [1] [1]
// CHECK: %[[SCALAR2:.*]] = tensor.collapse_shape %[[SLICE2]] []
// CHECK: %[[SIZE2:.*]] = hip.readback_scalar(%[[CTX]], %[[SCALAR2]] : tensor<i64>) -> i64
// CHECK: %[[INFERRED:.*]] = arith.divsi
// CHECK: arith.cmpi eq, %[[SIZE0]],
// CHECK: %[[RESOLVED0:.*]] = arith.select %{{.*}}, %[[INFERRED]], %[[SIZE0]] : i64
// CHECK: arith.cmpi eq, %[[SIZE1]],
// CHECK: %[[RESOLVED1:.*]] = arith.select %{{.*}}, %[[INFERRED]], %[[SIZE1]] : i64
// CHECK: arith.cmpi eq, %[[SIZE2]],
// CHECK: %[[RESOLVED2:.*]] = arith.select %{{.*}}, %[[INFERRED]], %[[SIZE2]] : i64
// CHECK: %[[TARGET:.*]] = tensor.from_elements %[[RESOLVED0]], %[[RESOLVED1]], %[[RESOLVED2]] : tensor<3xi64>
// CHECK: %[[RESULT:.*]] = tensor.reshape %[[DATA]](%[[TARGET]])
// CHECK: return %[[RESULT]]
// SCALARS-LABEL: func.func @unknown_shape_join(
// SCALARS-COUNT-3: hip.readback_scalar
// SCALARS-NOT: hip.readback_scalar
// SCALARS: return
func.func @unknown_shape_join(%data: tensor<?x?xf16>, %unknown: tensor<3xi64>, %condition: i1)
    -> tensor<?x?x?xf16> {
  %shape = "onnx.Shape"(%data) : (tensor<?x?xf16>) -> tensor<2xi64>
  %one = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %known = "onnx.Concat"(%shape, %one) {axis = 0 : si64}
    : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
  cf.cond_br %condition, ^left(%known : tensor<3xi64>), ^right(%unknown : tensor<3xi64>)
^left(%left_shape: tensor<3xi64>):
  cf.br ^join(%left_shape : tensor<3xi64>)
^right(%right_shape: tensor<3xi64>):
  cf.br ^join(%right_shape : tensor<3xi64>)
^join(%joined_shape: tensor<3xi64>):
  %result = "onnx.Reshape"(%data, %joined_shape) {allowzero = 1 : si64}
    : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
  return %result : tensor<?x?x?xf16>
}
