// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip \
// RUN:   --canonicalize --cse --hip-resolve-tensor-dims | FileCheck %s
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip \
// RUN:   | FileCheck %s --check-prefix=RAW

module {
  func.func @main_graph(%arg0: tensor<1xf16>) -> tensor<1xf16> {
    return %arg0 : tensor<1xf16>
  }

  // Shape/Gather/Unsqueeze/Concat is proven while generic ONNX constants are
  // still inline. Reshape consumes the scalar SSA directly and emits no
  // synchronized payload readback.
  // CHECK-LABEL: func.func @proven_shape_chain
  // CHECK-SAME: %[[DATA:[A-Za-z0-9_]+]]: tensor<?x?x?xf16>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[D0:.*]] = tensor.dim %[[DATA]], %[[C0]]
  // CHECK: %[[D0I:.*]] = arith.index_cast %[[D0]]
  // CHECK: %[[D1:.*]] = tensor.dim %[[DATA]], %[[C1]]
  // CHECK: %[[D1I:.*]] = arith.index_cast %[[D1]]
  // CHECK: %[[TARGET:.*]] = tensor.from_elements %[[D0I]], %[[D1I]],
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape %[[DATA]](%[[TARGET]])
  func.func @proven_shape_chain(
      %data: tensor<?x?x?xf16>) -> tensor<?x?x?x?xf16> {
    %idx0 = "onnx.Constant"() {value = dense<0> : tensor<i64>}
      : () -> tensor<i64>
    %idx1 = "onnx.Constant"() {value = dense<1> : tensor<i64>}
      : () -> tensor<i64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %c16 = "onnx.Constant"() {value = dense<16> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %c72 = "onnx.Constant"() {value = dense<72> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?x?xf16>) -> tensor<3xi64>
    %d0 = "onnx.Gather"(%shape, %idx0) {axis = 0 : si64}
      : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %d1 = "onnx.Gather"(%shape, %idx1) {axis = 0 : si64}
      : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %u0 = "onnx.Unsqueeze"(%d0, %axes)
      : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %u1 = "onnx.Unsqueeze"(%d1, %axes)
      : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %target = "onnx.Concat"(%u0, %u1, %c16, %c72) {axis = 0 : si64}
      : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<4xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?x?xf16>, tensor<4xi64>) -> tensor<?x?x?x?xf16>
    return %result : tensor<?x?x?x?xf16>
  }

  // Reordered Gather inputs must remain reordered in the proven target.
  // CHECK-LABEL: func.func @permuted_shape_chain
  // CHECK-SAME: %[[PERM_DATA:[A-Za-z0-9_]+]]: tensor<?x?x?xf16>
  // CHECK-DAG: %[[PERM_C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[PERM_C1:.*]] = arith.constant 1 : index
  // CHECK: %[[PERM_D1:.*]] = tensor.dim %[[PERM_DATA]], %[[PERM_C1]]
  // CHECK: %[[PERM_D1I:.*]] = arith.index_cast %[[PERM_D1]]
  // CHECK: %[[PERM_D0:.*]] = tensor.dim %[[PERM_DATA]], %[[PERM_C0]]
  // CHECK: %[[PERM_D0I:.*]] = arith.index_cast %[[PERM_D0]]
  // CHECK: tensor.from_elements %[[PERM_D1I]], %[[PERM_D0I]],
  func.func @permuted_shape_chain(
      %data: tensor<?x?x?xf16>) -> tensor<?x?x?x?xf16> {
    %idx0 = "onnx.Constant"() {value = dense<0> : tensor<i64>}
      : () -> tensor<i64>
    %idx1 = "onnx.Constant"() {value = dense<1> : tensor<i64>}
      : () -> tensor<i64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %c16 = "onnx.Constant"() {value = dense<16> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %c72 = "onnx.Constant"() {value = dense<72> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?x?xf16>) -> tensor<3xi64>
    %d0 = "onnx.Gather"(%shape, %idx0) {axis = 0 : si64}
      : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %d1 = "onnx.Gather"(%shape, %idx1) {axis = 0 : si64}
      : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %u0 = "onnx.Unsqueeze"(%d0, %axes)
      : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %u1 = "onnx.Unsqueeze"(%d1, %axes)
      : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %target = "onnx.Concat"(%u1, %u0, %c16, %c72) {axis = 0 : si64}
      : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<4xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 1 : si64}
      : (tensor<?x?x?xf16>, tensor<4xi64>) -> tensor<?x?x?x?xf16>
    return %result : tensor<?x?x?x?xf16>
  }

  // An opaque shape tensor argument has no host-side provenance. The fallback
  // retains one synchronized readback per output dimension.
  // CHECK-LABEL: func.func @unknown_shape_argument
  // CHECK-COUNT-4: hip.readback_scalar
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @unknown_shape_argument(
      %data: tensor<?x?x?xf16>,
      %shape: tensor<4xi64>) -> tensor<?x?x?x?xf16> {
    %result = "onnx.Reshape"(%data, %shape) {allowzero = 0 : si64}
      : (tensor<?x?x?xf16>, tensor<4xi64>) -> tensor<?x?x?x?xf16>
    return %result : tensor<?x?x?x?xf16>
  }

  // A dense carrier is a compile-time target at the Reshape conversion
  // boundary. It bypasses both provenance materialization and synchronized
  // payload readback; conversion never needs to inspect a memref global.
  // RAW-LABEL: func.func @dense_carrier_target
  // RAW-NOT: memref.get_global
  // RAW-NOT: hip.readback_scalar
  // RAW: %[[DENSE_TARGET:.*]] = tensor.from_elements
  // RAW-NEXT: tensor.reshape %{{.*}}(%[[DENSE_TARGET]])
  // CHECK-LABEL: func.func @dense_carrier_target
  // CHECK-NOT: memref.get_global
  // CHECK-NOT: hip.readback_scalar
  // CHECK: %[[DENSE_TARGET:.*]] = tensor.from_elements
  // CHECK-NEXT: tensor.reshape %{{.*}}(%[[DENSE_TARGET]])
  func.func @dense_carrier_target(
      %data: tensor<?x?x?xf16>) -> tensor<?x?xf16> {
    %target = hip.constant {
      value = dense<[0, -1]> : tensor<2xi64>
    } : tensor<2xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?x?xf16>, tensor<2xi64>) -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }

  // Provenance must not flatten a multidimensional Gather source. The outer
  // Reshape therefore retains synchronized fallback reads for both target
  // entries.
  // CHECK-LABEL: func.func @multidimensional_gather_fallback
  // CHECK-COUNT-2: hip.readback_scalar
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @multidimensional_gather_fallback(
      %data: tensor<?x?x?x?xf16>) -> tensor<?x?xf16> {
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?x?x?xf16>) -> tensor<4xi64>
    %shape2x2 = "onnx.Constant"() {
      value = dense<[2, 2]> : tensor<2xi64>
    } : () -> tensor<2xi64>
    %matrix = "onnx.Reshape"(%shape, %shape2x2) {allowzero = 0 : si64}
      : (tensor<4xi64>, tensor<2xi64>) -> tensor<2x2xi64>
    %row = "onnx.Constant"() {value = dense<1> : tensor<i64>}
      : () -> tensor<i64>
    %target = "onnx.Gather"(%matrix, %row) {axis = 0 : si64}
      : (tensor<2x2xi64>, tensor<i64>) -> tensor<2xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?x?x?xf16>, tensor<2xi64>) -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }

  // Rank-1 Slice preserves shape-vector order and remains provable.
  // CHECK-LABEL: func.func @rank_one_slice_provenance
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.from_elements
  // CHECK: tensor.reshape
  func.func @rank_one_slice_provenance(
      %data: tensor<?x?x?x?xf16>) -> tensor<?x?x?xf16> {
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?x?x?xf16>) -> tensor<4xi64>
    %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %one = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %slice = "onnx.Slice"(%shape, %starts, %ends, %axes)
      : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<2xi64>
    %target = "onnx.Concat"(%slice, %one) {axis = 0 : si64}
      : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 1 : si64}
      : (tensor<?x?x?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }

  // Slice of a rank-2 shape matrix is not flattened by provenance analysis.
  // CHECK-LABEL: func.func @multidimensional_slice_fallback
  // CHECK-COUNT-2: hip.readback_scalar
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @multidimensional_slice_fallback(
      %data: tensor<?x?x?x?xf16>) -> tensor<?x?xf16> {
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?x?x?xf16>) -> tensor<4xi64>
    %shape2x2 = "onnx.Constant"() {
      value = dense<[2, 2]> : tensor<2xi64>
    } : () -> tensor<2xi64>
    %matrix = "onnx.Reshape"(%shape, %shape2x2) {allowzero = 0 : si64}
      : (tensor<4xi64>, tensor<2xi64>) -> tensor<2x2xi64>
    %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %row = "onnx.Slice"(%matrix, %starts, %ends, %axes)
      : (tensor<2x2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<1x2xi64>
    %vectorShape = "onnx.Constant"() {
      value = dense<2> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %target = "onnx.Reshape"(%row, %vectorShape) {allowzero = 0 : si64}
      : (tensor<1x2xi64>, tensor<1xi64>) -> tensor<2xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?x?x?xf16>, tensor<2xi64>) -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }

  // Extreme negative/positive indices are clipped without signed overflow.
  // CHECK-LABEL: func.func @extreme_shape_slice_indices
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @extreme_shape_slice_indices(
      %data: tensor<?x?x?xf16>) -> tensor<?x?x?x?xf16> {
    %shape = "onnx.Shape"(%data) {
      start = -9223372036854775808 : si64,
      end = 9223372036854775807 : si64
    } : (tensor<?x?x?xf16>) -> tensor<3xi64>
    %starts = "onnx.Constant"() {
      value = dense<-9223372036854775808> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {
      value = dense<9223372036854775807> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %slice = "onnx.Slice"(%shape, %starts, %ends, %axes)
      : (tensor<3xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<3xi64>
    %one = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %target = "onnx.Concat"(%slice, %one) {axis = 0 : si64}
      : (tensor<3xi64>, tensor<1xi64>) -> tensor<4xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 1 : si64}
      : (tensor<?x?x?xf16>, tensor<4xi64>) -> tensor<?x?x?x?xf16>
    return %result : tensor<?x?x?x?xf16>
  }

  // Plans for dependent Reshapes are all queried before the first
  // materialization mutates IR.
  // RAW-LABEL: func.func @dependent_reshapes
  // RAW-NOT: hip.readback_scalar
  // RAW-COUNT-2: tensor.expand_shape
  func.func @dependent_reshapes(
      %data: tensor<?x?xf16>) -> tensor<?x1x?x1xf16> {
    %idx0 = "onnx.Constant"() {value = dense<0> : tensor<i64>}
      : () -> tensor<i64>
    %idx1 = "onnx.Constant"() {value = dense<1> : tensor<i64>}
      : () -> tensor<i64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %one = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %sourceShape = "onnx.Shape"(%data)
      : (tensor<?x?xf16>) -> tensor<2xi64>
    %d0 = "onnx.Gather"(%sourceShape, %idx0) {axis = 0 : si64}
      : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    %d1 = "onnx.Gather"(%sourceShape, %idx1) {axis = 0 : si64}
      : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    %u0 = "onnx.Unsqueeze"(%d0, %axes)
      : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %u1 = "onnx.Unsqueeze"(%d1, %axes)
      : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %target0 = "onnx.Concat"(%u0, %one, %u1) {axis = 0 : si64}
      : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %first = "onnx.Reshape"(%data, %target0) {allowzero = 1 : si64}
      : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x1x?xf16>
    %firstShape = "onnx.Shape"(%first)
      : (tensor<?x1x?xf16>) -> tensor<3xi64>
    %target1 = "onnx.Concat"(%firstShape, %one) {axis = 0 : si64}
      : (tensor<3xi64>, tensor<1xi64>) -> tensor<4xi64>
    %second = "onnx.Reshape"(%first, %target1) {allowzero = 1 : si64}
      : (tensor<?x1x?xf16>, tensor<4xi64>) -> tensor<?x1x?x1xf16>
    return %second : tensor<?x1x?x1xf16>
  }

  // Add(MatMul(x, W), bias) preserves x's leading dimensions. The proven
  // input-dimension map removes allowzero's redundant zero-select.
  // CHECK-LABEL: func.func @matmul_add_equivalence
  // CHECK-NOT: hip.readback_scalar
  // CHECK-NOT: arith.select
  // CHECK: tensor.reshape
  func.func @matmul_add_equivalence(
      %data: tensor<?x?x4xf16>,
      %weight: tensor<4x4xf16>,
      %bias: tensor<4xf16>) -> tensor<?x?x?x?xf16> {
    %mm = "onnx.MatMul"(%data, %weight)
      : (tensor<?x?x4xf16>, tensor<4x4xf16>) -> tensor<?x?x4xf16>
    %sum = "onnx.Add"(%bias, %mm)
      : (tensor<4xf16>, tensor<?x?x4xf16>) -> tensor<?x?x4xf16>
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?x4xf16>) -> tensor<3xi64>
    %idx0 = "onnx.Constant"() {value = dense<0> : tensor<i64>}
      : () -> tensor<i64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %two = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %d0 = "onnx.Gather"(%shape, %idx0) {axis = 0 : si64}
      : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %u0 = "onnx.Unsqueeze"(%d0, %axes)
      : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %target = "onnx.Concat"(%u0, %two, %two, %two) {axis = 0 : si64}
      : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<4xi64>
    %result = "onnx.Reshape"(%sum, %target) {allowzero = 0 : si64}
      : (tensor<?x?x4xf16>, tensor<4xi64>) -> tensor<?x?x?x?xf16>
    return %result : tensor<?x?x?x?xf16>
  }

  // A CFG join erases the first MatMul operand's batch-dimension root. The
  // second operand's non-unit dynamic root must not overwrite that unknown:
  // the original target is based on %rhs, not on the MatMul result.
  // RAW-LABEL: func.func @matmul_unknown_batch_is_ambiguous
  // RAW-SAME: %[[RHS:[A-Za-z0-9_]+]]: tensor<?x3x4xf16>
  // RAW: %[[MM:.*]] = hip.matmul
  // RAW-NEXT: %[[TARGET_DIM:.*]] = tensor.dim %[[RHS]],
  // RAW-NEXT: %[[TARGET_DIM_I64:.*]] = arith.index_cast %[[TARGET_DIM]]
  // RAW-NOT: hip.readback_scalar
  // RAW: tensor.reshape %[[MM]]
  func.func @matmul_unknown_batch_is_ambiguous(
      %lhs0: tensor<?x2x3xf16>,
      %lhs1: tensor<?x2x3xf16>,
      %rhs: tensor<?x3x4xf16>,
      %condition: i1) -> tensor<?x?xf16> {
    cf.cond_br %condition, ^left(%lhs0 : tensor<?x2x3xf16>),
      ^right(%lhs1 : tensor<?x2x3xf16>)
  ^left(%leftValue: tensor<?x2x3xf16>):
    cf.br ^join(%leftValue : tensor<?x2x3xf16>)
  ^right(%rightValue: tensor<?x2x3xf16>):
    cf.br ^join(%rightValue : tensor<?x2x3xf16>)
  ^join(%lhs: tensor<?x2x3xf16>):
    %mm = "onnx.MatMul"(%lhs, %rhs)
      : (tensor<?x2x3xf16>, tensor<?x3x4xf16>) -> tensor<?x2x4xf16>
    %c0 = arith.constant 0 : index
    %rhsBatch = tensor.dim %rhs, %c0 : tensor<?x3x4xf16>
    %rhsBatchI64 = arith.index_cast %rhsBatch : index to i64
    %minusOne = arith.constant -1 : i64
    %target = tensor.from_elements %rhsBatchI64, %minusOne : tensor<2xi64>
    %result = "onnx.Reshape"(%mm, %target) {allowzero = 0 : si64}
      : (tensor<?x2x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }

  // Cast preserves semantic dimensions. Canonicalizing those facts lets the
  // materializer rewrite the proof onto the Reshape input itself.
  // CHECK-LABEL: func.func @cast_dimension_equivalence
  // CHECK-NOT: hip.readback_scalar
  // CHECK-NOT: arith.select
  // CHECK: tensor.reshape
  func.func @cast_dimension_equivalence(
      %data: tensor<?x?xf16>) -> tensor<?x4xf32> {
    %cast = "onnx.Cast"(%data) {to = f32}
      : (tensor<?x?xf16>) -> tensor<?x?xf32>
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?xf16>) -> tensor<2xi64>
    %result = "onnx.Reshape"(%cast, %shape) {allowzero = 0 : si64}
      : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x4xf32>
    return %result : tensor<?x4xf32>
  }

  // Two dynamic Add operands are ambiguous: the correct broadcast select must
  // remain because neither dimension is proven to be the canonical source.
  // CHECK-LABEL: func.func @ambiguous_add_keeps_select
  // CHECK-NOT: hip.readback_scalar
  // CHECK: arith.select
  // CHECK: tensor.reshape
  func.func @ambiguous_add_keeps_select(
      %lhs: tensor<?x?x4xf16>,
      %rhs: tensor<?x?x4xf16>) -> tensor<?x?x?x?xf16> {
    %sum = "onnx.Add"(%lhs, %rhs)
      : (tensor<?x?x4xf16>, tensor<?x?x4xf16>) -> tensor<?x?x4xf16>
    %shape = "onnx.Shape"(%lhs)
      : (tensor<?x?x4xf16>) -> tensor<3xi64>
    %idx0 = "onnx.Constant"() {value = dense<0> : tensor<i64>}
      : () -> tensor<i64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %two = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %d0 = "onnx.Gather"(%shape, %idx0) {axis = 0 : si64}
      : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %u0 = "onnx.Unsqueeze"(%d0, %axes)
      : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %target = "onnx.Concat"(%u0, %two, %two, %two) {axis = 0 : si64}
      : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<4xi64>
    %result = "onnx.Reshape"(%sum, %target) {allowzero = 0 : si64}
      : (tensor<?x?x4xf16>, tensor<4xi64>) -> tensor<?x?x?x?xf16>
    return %result : tensor<?x?x?x?xf16>
  }

  // Repeated SSA operands carry identical dimension facts and are not
  // ambiguous dynamic broadcast sources.
  // CHECK-LABEL: func.func @repeated_add_reuses_dimension
  // CHECK-NOT: arith.select
  // CHECK: tensor.reshape
  func.func @repeated_add_reuses_dimension(
      %data: tensor<?x?x4xf16>) -> tensor<?x?x?x?xf16> {
    %sum = "onnx.Add"(%data, %data)
      : (tensor<?x?x4xf16>, tensor<?x?x4xf16>) -> tensor<?x?x4xf16>
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?x4xf16>) -> tensor<3xi64>
    %idx0 = "onnx.Constant"() {value = dense<0> : tensor<i64>}
      : () -> tensor<i64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %two = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %d0 = "onnx.Gather"(%shape, %idx0) {axis = 0 : si64}
      : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %u0 = "onnx.Unsqueeze"(%d0, %axes)
      : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %target = "onnx.Concat"(%u0, %two, %two, %two) {axis = 0 : si64}
      : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<4xi64>
    %result = "onnx.Reshape"(%sum, %target) {allowzero = 0 : si64}
      : (tensor<?x?x4xf16>, tensor<4xi64>) -> tensor<?x?x?x?xf16>
    return %result : tensor<?x?x?x?xf16>
  }

  // allowzero=0 copies the corresponding input extent instead of preserving
  // the literal zero.
  // CHECK-LABEL: func.func @allowzero_zero_copies_input
  // CHECK-SAME: %[[ZERO_DATA:[A-Za-z0-9_]+]]: tensor<?x?xf16>
  // CHECK: %[[ZERO_D0:.*]] = tensor.dim %[[ZERO_DATA]]
  // CHECK: %[[ZERO_D0I:.*]] = arith.index_cast %[[ZERO_D0]]
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.from_elements %[[ZERO_D0I]],
  // CHECK: tensor.reshape
  func.func @allowzero_zero_copies_input(
      %data: tensor<?x?xf16>) -> tensor<?x1xf16> {
    %target = "onnx.Constant"() {
      value = dense<[0, 1]> : tensor<2xi64>
    } : () -> tensor<2xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?xf16>, tensor<2xi64>) -> tensor<?x1xf16>
    return %result : tensor<?x1xf16>
  }

  // allowzero=1 preserves a literal zero.
  // CHECK-LABEL: func.func @allowzero_one_preserves_zero
  // CHECK-NOT: hip.readback_scalar
  // CHECK: %[[ZERO_TARGET:.*]] = arith.constant dense<[0, 1]> : tensor<2xi64>
  // CHECK: tensor.reshape %{{.*}}(%[[ZERO_TARGET]])
  func.func @allowzero_one_preserves_zero(
      %data: tensor<?x?xf16>) -> tensor<?x1xf16> {
    %target = "onnx.Constant"() {
      value = dense<[0, 1]> : tensor<2xi64>
    } : () -> tensor<2xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 1 : si64}
      : (tensor<?x?xf16>, tensor<2xi64>) -> tensor<?x1xf16>
    return %result : tensor<?x1xf16>
  }

  // Narrowing index casts can wrap a nonnegative dimension to -1. They must
  // not receive the nonnegative fast-path proof, so generic -1 handling stays.
  // CHECK-LABEL: func.func @narrowing_index_cast_keeps_inference
  // CHECK: arith.cmpi eq
  // CHECK: arith.select
  // CHECK: tensor.reshape
  func.func @narrowing_index_cast_keeps_inference(
      %data: tensor<?x?xf16>) -> tensor<?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %data, %c0 : tensor<?x?xf16>
    %narrow = arith.index_cast %dim : index to i32
    %wide = arith.extsi %narrow : i32 to i64
    %c1 = arith.constant 1 : i64
    %target = tensor.from_elements %wide, %c1, %c1 : tensor<3xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }

  // An ONNX literal becomes a dense carrier before Reshape conversion.
  // Conversion rematerializes its entries without provenance or readback,
  // including -1 inference and allowzero=0.
  // CHECK-LABEL: func.func @onnx_literal_carrier_target
  // CHECK-NOT: hip.readback_scalar
  // CHECK: arith.cmpi eq
  // CHECK: arith.select
  // CHECK: tensor.reshape
  func.func @onnx_literal_carrier_target(
      %data: tensor<?x?x?xf16>) -> tensor<?x3x?xf16> {
    %target = "onnx.Constant"() {
      value = dense<[0, 3, -1]> : tensor<3xi64>
    } : () -> tensor<3xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?x?xf16>, tensor<3xi64>) -> tensor<?x3x?xf16>
    return %result : tensor<?x3x?xf16>
  }

  // A proven vector containing -1 bypasses payload readback but retains the
  // existing inferred-dimension arithmetic.
  // CHECK-LABEL: func.func @proven_minus_one
  // CHECK-NOT: hip.readback_scalar
  // CHECK: arith.cmpi eq
  // CHECK: arith.select
  // CHECK: tensor.reshape
  func.func @proven_minus_one(
      %data: tensor<?x?x?xf16>) -> tensor<?x?x?x?xf16> {
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?x?xf16>) -> tensor<3xi64>
    %idx0 = "onnx.Constant"() {value = dense<0> : tensor<i64>}
      : () -> tensor<i64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %minusOne = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %c16 = "onnx.Constant"() {value = dense<16> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %c72 = "onnx.Constant"() {value = dense<72> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %d0 = "onnx.Gather"(%shape, %idx0) {axis = 0 : si64}
      : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %u0 = "onnx.Unsqueeze"(%d0, %axes)
      : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %target = "onnx.Concat"(%u0, %minusOne, %c16, %c72) {axis = 0 : si64}
      : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<4xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?x?xf16>, tensor<4xi64>) -> tensor<?x?x?x?xf16>
    return %result : tensor<?x?x?x?xf16>
  }

  // Identical positive constants reaching a CFG block argument retain their
  // interned positive proof. The joined SSA value is not a compile-time tensor
  // constant, so avoiding readback uniquely requires provenance.
  // CHECK-LABEL: func.func @allowzero_minus_one_positive_join
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @allowzero_minus_one_positive_join(
      %data: tensor<?x?x?xf16>,
      %condition: i1) -> tensor<2x?x3xf16> {
    %two = arith.constant 2 : i64
    cf.cond_br %condition, ^left(%two : i64), ^right(%two : i64)
  ^left(%leftValue: i64):
    cf.br ^join(%leftValue : i64)
  ^right(%rightValue: i64):
    cf.br ^join(%rightValue : i64)
  ^join(%positive: i64):
    %minusOne = arith.constant -1 : i64
    %three = arith.constant 3 : i64
    %target =
      tensor.from_elements %positive, %minusOne, %three : tensor<3xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 1 : si64}
      : (tensor<?x?x?xf16>, tensor<3xi64>) -> tensor<2x?x3xf16>
    return %result : tensor<2x?x3xf16>
  }

  // A tensor dimension is nonnegative but not strictly positive. Combined
  // with allowzero=1 and -1, provenance must refuse the marker and retain the
  // synchronized fallback.
  // CHECK-LABEL: func.func @allowzero_minus_one_nonnegative_fallback
  // CHECK-COUNT-3: hip.readback_scalar
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @allowzero_minus_one_nonnegative_fallback(
      %data: tensor<?x?x?xf16>) -> tensor<?x?x3xf16> {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %data, %c0 : tensor<?x?x?xf16>
    %d0i64 = arith.index_cast %d0 : index to i64
    %minusOne = arith.constant -1 : i64
    %three = arith.constant 3 : i64
    %target = tensor.from_elements %d0i64, %minusOne, %three : tensor<3xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 1 : si64}
      : (tensor<?x?x?xf16>, tensor<3xi64>) -> tensor<?x?x3xf16>
    return %result : tensor<?x?x3xf16>
  }

  // A scalar constant wider than i64 cannot be represented by a provenance
  // constant. Joining it with an opaque scalar and narrowing it into a target
  // shape must safely remain unknown instead of asserting in APInt accessors.
  // CHECK-LABEL: func.func @wide_integer_constant_fallback
  // CHECK: arith.constant 1208925819614629174706176 : i128
  // CHECK-COUNT-3: hip.readback_scalar
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @wide_integer_constant_fallback(
      %data: tensor<?x?x?xf16>,
      %opaque: i128,
      %condition: i1) -> tensor<?x?x3xf16> {
    %wide = arith.constant 1208925819614629174706176 : i128
    cf.cond_br %condition, ^constant(%wide : i128), ^opaque(%opaque : i128)
  ^constant(%constantValue: i128):
    cf.br ^join(%constantValue : i128)
  ^opaque(%opaqueValue: i128):
    cf.br ^join(%opaqueValue : i128)
  ^join(%joined: i128):
    %narrow = arith.trunci %joined : i128 to i64
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %target = tensor.from_elements %narrow, %one, %three : tensor<3xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?x?xf16>, tensor<3xi64>) -> tensor<?x?x3xf16>
    return %result : tensor<?x?x3xf16>
  }

  // Static reshapes cannot consume provenance and skip whole-function
  // dataflow, even when the function contains other integer SSA.
  // CHECK-LABEL: func.func @static_reshape_skips_provenance
  // CHECK: arith.constant 1208925819614629174706176 : i128
  // CHECK-NOT: hip.readback_scalar
  func.func @static_reshape_skips_provenance(
      %data: tensor<2x2xf16>) -> (tensor<4xf16>, i128) {
    %wide = arith.constant 1208925819614629174706176 : i128
    %target = "onnx.Constant"() {
      value = dense<4> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<2x2xf16>, tensor<1xi64>) -> tensor<4xf16>
    return %result, %wide : tensor<4xf16>, i128
  }

  // Materialization updates only the Reshape operand. The complete proven
  // vector remains available to an unrelated non-Reshape consumer.
  // RAW-LABEL: func.func @partial_shape_user
  // RAW: %[[FULL:.*]] = tensor.from_elements
  // RAW: %[[RESHAPE_SHAPE:.*]] = tensor.from_elements
  // RAW: tensor.reshape %{{.*}}(%[[RESHAPE_SHAPE]])
  // RAW: call @consume_shape(%{{.*}}, %[[FULL]])
  func.func @partial_shape_user(
      %data: tensor<?x?xf16>) -> tensor<?x1xf16> {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %data, %c0 : tensor<?x?xf16>
    %d0i64 = arith.index_cast %d0 : index to i64
    %one = arith.constant 1 : i64
    %seven = arith.constant 7 : i64
    %full = tensor.from_elements %d0i64, %one, %seven : tensor<3xi64>
    %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %slice = "onnx.Slice"(%full, %starts, %ends, %axes)
      : (tensor<3xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<2xi64>
    %result = "onnx.Reshape"(%data, %slice) {allowzero = 1 : si64}
      : (tensor<?x?xf16>, tensor<2xi64>) -> tensor<?x1xf16>
    func.call @consume_shape(%full) : (tensor<3xi64>) -> ()
    return %result : tensor<?x1xf16>
  }

  // A shared producer subgraph is analyzed once and both consumers receive
  // the same payload fact without synchronized readback.
  // CHECK-LABEL: func.func @shared_shape_subgraph
  // CHECK-NOT: hip.readback_scalar
  // CHECK-COUNT-2: tensor.reshape
  func.func @shared_shape_subgraph(
      %lhs: tensor<?x?xf16>,
      %rhs: tensor<?x?xf16>) -> (tensor<?x4xf16>, tensor<?x4xf16>) {
    %shape = "onnx.Shape"(%lhs)
      : (tensor<?x?xf16>) -> tensor<2xi64>
    %lhsResult = "onnx.Reshape"(%lhs, %shape) {allowzero = 1 : si64}
      : (tensor<?x?xf16>, tensor<2xi64>) -> tensor<?x4xf16>
    %rhsResult = "onnx.Reshape"(%rhs, %shape) {allowzero = 1 : si64}
      : (tensor<?x?xf16>, tensor<2xi64>) -> tensor<?x4xf16>
    return %lhsResult, %rhsResult : tensor<?x4xf16>, tensor<?x4xf16>
  }

  // Identical payloads reaching a block argument from distinct CFG
  // predecessors survive the lattice join.
  // CHECK-LABEL: func.func @identical_dataflow_join
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @identical_dataflow_join(
      %data: tensor<?x?xf16>,
      %condition: i1) -> tensor<?x4xf16> {
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?xf16>) -> tensor<2xi64>
    cf.cond_br %condition, ^left(%shape : tensor<2xi64>),
      ^right(%shape : tensor<2xi64>)
  ^left(%leftShape: tensor<2xi64>):
    cf.br ^join(%leftShape : tensor<2xi64>)
  ^right(%rightShape: tensor<2xi64>):
    cf.br ^join(%rightShape : tensor<2xi64>)
  ^join(%joinedShape: tensor<2xi64>):
    %result = "onnx.Reshape"(%data, %joinedShape) {allowzero = 1 : si64}
      : (tensor<?x?xf16>, tensor<2xi64>) -> tensor<?x4xf16>
    return %result : tensor<?x4xf16>
  }

  // Divergent payloads conservatively join to unknown and retain synchronized
  // fallback reads.
  // CHECK-LABEL: func.func @divergent_dataflow_join
  // CHECK-COUNT-2: hip.readback_scalar
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @divergent_dataflow_join(
      %data: tensor<?x?xf16>,
      %condition: i1) -> tensor<?x4xf16> {
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?xf16>) -> tensor<2xi64>
    %ones = "onnx.Constant"() {
      value = dense<1> : tensor<2xi64>
    } : () -> tensor<2xi64>
    cf.cond_br %condition, ^left(%shape : tensor<2xi64>),
      ^right(%ones : tensor<2xi64>)
  ^left(%leftShape: tensor<2xi64>):
    cf.br ^join(%leftShape : tensor<2xi64>)
  ^right(%rightShape: tensor<2xi64>):
    cf.br ^join(%rightShape : tensor<2xi64>)
  ^join(%joinedShape: tensor<2xi64>):
    %result = "onnx.Reshape"(%data, %joinedShape) {allowzero = 1 : si64}
      : (tensor<?x?xf16>, tensor<2xi64>) -> tensor<?x4xf16>
    return %result : tensor<?x4xf16>
  }

  // One proven and one unknown incoming payload also joins to unknown.
  // CHECK-LABEL: func.func @interesting_unknown_dataflow_join
  // CHECK-COUNT-2: hip.readback_scalar
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @interesting_unknown_dataflow_join(
      %data: tensor<?x?xf16>,
      %unknown: tensor<2xi64>,
      %condition: i1) -> tensor<?x4xf16> {
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?xf16>) -> tensor<2xi64>
    cf.cond_br %condition, ^left(%shape : tensor<2xi64>),
      ^right(%unknown : tensor<2xi64>)
  ^left(%leftShape: tensor<2xi64>):
    cf.br ^join(%leftShape : tensor<2xi64>)
  ^right(%rightShape: tensor<2xi64>):
    cf.br ^join(%rightShape : tensor<2xi64>)
  ^join(%joinedShape: tensor<2xi64>):
    %result = "onnx.Reshape"(%data, %joinedShape) {allowzero = 1 : si64}
      : (tensor<?x?xf16>, tensor<2xi64>) -> tensor<?x4xf16>
    return %result : tensor<?x4xf16>
  }

  // hip.loop has an outlined body whose result dimensions cannot be inferred
  // from v_init alone. Keep the target rooted at %initial instead of claiming
  // equivalence with the loop result.
  // RAW-LABEL: func.func @direct_hip_loop_no_propagation
  // RAW-SAME: %[[INITIAL:[A-Za-z0-9_]+]]: tensor<?x4xf16>
  // RAW: %[[LOOP:.*]] = hip.loop
  // RAW: %[[INITIAL_DIM:.*]] = tensor.dim %[[INITIAL]],
  // RAW-NEXT: %[[INITIAL_DIM_I64:.*]] = arith.index_cast %[[INITIAL_DIM]]
  // RAW-NEXT: %[[LOOP_TARGET:.*]] = tensor.from_elements %[[INITIAL_DIM_I64]],
  // RAW: tensor.reshape %[[LOOP]](%[[LOOP_TARGET]])
  func.func @direct_hip_loop_no_propagation(
      %ctx: !hip.context,
      %tripCount: index,
      %condition: i1,
      %initial: tensor<?x4xf16>) -> tensor<?x?xf16> {
    %loop = hip.loop(%ctx, %tripCount, %condition)
      iter_args(%initial : tensor<?x4xf16>)
      -> (tensor<?x4xf16>)
      body @direct_loop_body
      {cond_is_passthrough, num_loop_carried = 1 : i32}
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %initial, %c0 : tensor<?x4xf16>
    %d0i64 = arith.index_cast %d0 : index to i64
    %c4 = arith.constant 4 : i64
    %target = tensor.from_elements %d0i64, %c4 : tensor<2xi64>
    %result = "onnx.Reshape"(%loop, %target) {allowzero = 1 : si64}
      : (tensor<?x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }

  // A divergent loop backedge must widen the iter-arg payload to unknown.
  // CHECK-LABEL: func.func @divergent_loop_backedge
  // CHECK-COUNT-3: hip.readback_scalar
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @divergent_loop_backedge(
      %data: tensor<?x?xf16>,
      %lb: index, %ub: index, %step: index) -> tensor<?x?x?xf16> {
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?xf16>) -> tensor<2xi64>
    %one = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %initial = "onnx.Concat"(%shape, %one) {axis = 0 : si64}
      : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
    %other = "onnx.Constant"() {
      value = dense<[1, 1, 1]> : tensor<3xi64>
    } : () -> tensor<3xi64>
    %joined = scf.for %iv = %lb to %ub step %step
        iter_args(%carried = %initial) -> tensor<3xi64> {
      scf.yield %other : tensor<3xi64>
    }
    %result = "onnx.Reshape"(%data, %joined) {allowzero = 1 : si64}
      : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }

  // Divergent RegionBranch results also join to unknown.
  // CHECK-LABEL: func.func @divergent_region_result
  // CHECK-COUNT-3: hip.readback_scalar
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.reshape
  func.func @divergent_region_result(
      %data: tensor<?x?xf16>,
      %condition: i1) -> tensor<?x?x?xf16> {
    %shape = "onnx.Shape"(%data)
      : (tensor<?x?xf16>) -> tensor<2xi64>
    %one = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
      : () -> tensor<1xi64>
    %proven = "onnx.Concat"(%shape, %one) {axis = 0 : si64}
      : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
    %other = "onnx.Constant"() {
      value = dense<[1, 1, 1]> : tensor<3xi64>
    } : () -> tensor<3xi64>
    %joined = scf.if %condition -> tensor<3xi64> {
      scf.yield %proven : tensor<3xi64>
    } else {
      scf.yield %other : tensor<3xi64>
    }
    %result = "onnx.Reshape"(%data, %joined) {allowzero = 1 : si64}
      : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }

  func.func private @consume_shape(tensor<3xi64>)
  func.func private @direct_loop_body(
      %ctx: !hip.context, %iter: tensor<i64>, %cond: tensor<i1>,
      %current: tensor<?x4xf16>, %frame: !hip.loop_frame)
      -> (i32, tensor<?x4xf16>) {
    %status = arith.constant 0 : i32
    return %status, %current : i32, tensor<?x4xf16>
  }
}
