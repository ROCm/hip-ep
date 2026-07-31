// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics -hipsr-populate-shape-region %s | FileCheck %s --check-prefix=POPULATE
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics -canonicalize %s | FileCheck %s --check-prefix=CANONICALIZE
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics -canonicalize -hipsr-populate-shape-region %s | FileCheck %s --check-prefix=FOLD-POPULATE

// Runtime-shape Expand uses the barrier layout: context, input tensor, and
// requested-shape tensor. The tensor-stage recipe extracts each extent.
// POPULATE-LABEL: func.func @expand_runtime(
// POPULATE-NEXT: %[[INIT:.+]] = hipsr.placeholder(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x3xf16>, tensor<2xi64>) {type = #hipsr.placeholder_type<barrier>} : tensor<?x?xf16> shape_region {
// POPULATE-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[INPUT:.+]]: tensor<?x3xf16>, %[[REQUEST:.+]]: tensor<2xi64>):
// POPULATE-NEXT: %[[INPUT_SHAPE:.+]] = shape.shape_of %[[INPUT]]
// POPULATE-NEXT: %[[INDEX0:.+]] = arith.constant 0 : index
// POPULATE-NEXT: %[[EXTENT0_I64:.+]] = tensor.extract %[[REQUEST]][%[[INDEX0]]]
// POPULATE-NEXT: %[[EXTENT0:.+]] = arith.index_cast %[[EXTENT0_I64]] : i64 to index
// POPULATE-NEXT: %[[INDEX1:.+]] = arith.constant 1 : index
// POPULATE-NEXT: %[[EXTENT1_I64:.+]] = tensor.extract %[[REQUEST]][%[[INDEX1]]]
// POPULATE-NEXT: %[[EXTENT1:.+]] = arith.index_cast %[[EXTENT1_I64]] : i64 to index
// POPULATE-NEXT: %[[REQUEST_SHAPE:.+]] = shape.from_extents %[[EXTENT0]], %[[EXTENT1]]
// POPULATE-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]]
// POPULATE-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// POPULATE-NEXT: %[[BROADCAST:.+]] = shape.broadcast %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]]
// POPULATE-NEXT: shape.assuming_yield %[[BROADCAST]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: %[[RESULT:.+]] = hipsr.expand
// POPULATE-NOT: shape_region
// POPULATE-NEXT: return %[[RESULT]] : tensor<?x?xf16>
func.func @expand_runtime(
    %ctx: !hipsr.context, %input: tensor<?x3xf16>, %shape: tensor<2xi64>)
    -> tensor<?x?xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
      {type = #hipsr.placeholder_type<barrier>} : tensor<?x?xf16>
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
      outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
  return %result : tensor<?x?xf16>
}

// -----

// Attribute Expand uses one normal input shape and materializes its extents.
// POPULATE-LABEL: func.func @expand_shape_attr(
// POPULATE-NEXT: %[[INIT:.+]] = hipsr.placeholder(%{{.+}}) ins(%{{.+}} : tensor<?x3xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<?x?xf16> shape_region {
// POPULATE-NEXT: ^bb0(%[[INPUT_SHAPE:.+]]: !shape.shape):
// POPULATE-NEXT: %[[EXTENT0:.+]] = arith.constant 4 : index
// POPULATE-NEXT: %[[EXTENT1:.+]] = arith.constant 3 : index
// POPULATE-NEXT: %[[REQUEST_SHAPE:.+]] = shape.from_extents %[[EXTENT0]], %[[EXTENT1]]
// POPULATE-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]]
// POPULATE-NEXT: %[[RESULT_SHAPE:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// POPULATE-NEXT: %[[BROADCAST:.+]] = shape.broadcast %[[INPUT_SHAPE]], %[[REQUEST_SHAPE]]
// POPULATE-NEXT: shape.assuming_yield %[[BROADCAST]] : !shape.shape
// POPULATE-NEXT: }
// POPULATE-NEXT: hipsr.shape_yield %[[RESULT_SHAPE]] : !shape.shape
func.func @expand_shape_attr(
    %ctx: !hipsr.context, %input: tensor<?x3xf16>) -> tensor<?x?xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x3xf16>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x?xf16>
  %result = hipsr.expand(%ctx)
      ins(%input : tensor<?x3xf16>)
      outs(%init : tensor<?x?xf16>)
      {shape_attr = array<i64: 4, 3>} : tensor<?x?xf16>
  return %result : tensor<?x?xf16>
}

// -----

// Canonicalization folds Expand and removes the placeholder's stale shape
// input in the same pass. Population later selects the normal layout.
// CANONICALIZE-LABEL: func.func @expand_arith_constant(
// CANONICALIZE-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x3xf16>)
// CANONICALIZE-NOT: tensor<2xi64>
// CANONICALIZE: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]])
// CANONICALIZE-SAME: ins(%[[INPUT]] : tensor<?x3xf16>)
// CANONICALIZE-SAME: #hipsr.placeholder_type<barrier>
// CANONICALIZE-NEXT: %[[RESULT:.+]] = hipsr.expand(%[[CTX]])
// CANONICALIZE-SAME: ins(%[[INPUT]] : tensor<?x3xf16>)
// CANONICALIZE-SAME: outs(%[[INIT]] : tensor<?x?xf16>)
// CANONICALIZE-SAME: {shape_attr = array<i64: 4, 3>}
// FOLD-POPULATE-LABEL: func.func @expand_arith_constant(
// FOLD-POPULATE: %[[INIT:.+]] = hipsr.placeholder
// FOLD-POPULATE-SAME: ins(%{{.+}} : tensor<?x3xf16>)
// FOLD-POPULATE-SAME: #hipsr.placeholder_type<normal>
// FOLD-POPULATE-SAME: shape_region {
// FOLD-POPULATE-NEXT: ^bb0(%{{.+}}: !shape.shape):
// FOLD-POPULATE: %[[RESULT:.+]] = hipsr.expand
// FOLD-POPULATE-SAME: ins(%{{.+}} : tensor<?x3xf16>)
// FOLD-POPULATE-SAME: {shape_attr = array<i64: 4, 3>}
func.func @expand_arith_constant(
    %ctx: !hipsr.context, %input: tensor<?x3xf16>) -> tensor<?x?xf16> {
  %shape = arith.constant dense<[4, 3]> : tensor<2xi64>
  %init = hipsr.placeholder(%ctx)
      ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
      {type = #hipsr.placeholder_type<barrier>} : tensor<?x?xf16>
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
      outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
  return %result : tensor<?x?xf16>
}

// -----

// Post-bufferization host-shape syntax remains valid, but this stage does not
// populate memref placeholders.
// POPULATE-LABEL: func.func @expand_host_shape_memref(
// POPULATE: hipsr.expand
// POPULATE-NOT: shape_region
// POPULATE-NEXT: return
func.func @expand_host_shape_memref(
    %ctx: !hipsr.context,
    %input: memref<?x3xf16, #hipsr.mem<device>>,
    %shape: memref<2xi64, #hipsr.mem<host>>,
    %init: memref<?x?xf16, #hipsr.mem<device>>) {
  hipsr.expand(%ctx)
      ins(%input, %shape : memref<?x3xf16, #hipsr.mem<device>>,
                            memref<2xi64, #hipsr.mem<host>>)
      outs(%init : memref<?x?xf16, #hipsr.mem<device>>)
  return
}

// -----

func.func @expand_both_shapes(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>, %shape: tensor<2xi64>,
    %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{cannot have both shape operand and shape_attr attribute}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<2xi64>)
      outs(%init : tensor<2x3xf16>)
      {shape_attr = array<i64: 2, 3>} : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

func.func @expand_missing_shape(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{must have either shape operand or shape_attr attribute}}
  %result = hipsr.expand(%ctx)
      ins(%input : tensor<2x3xf16>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

func.func @expand_shape_rank(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %shape: tensor<1x2xi64>, %init: tensor<2x3xf16>)
    -> tensor<2x3xf16> {
  // expected-error@+1 {{shape must be rank-1}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<1x2xi64>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

func.func @expand_shape_element_type(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>, %shape: tensor<2xi32>,
    %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{shape element type must be i64}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<2xi32>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

func.func @expand_dynamic_shape_length(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>, %shape: tensor<?xi64>,
    %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{shape length must be static}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<?xi64>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

// Expand preserves the input element type.
func.func @expand_element_mismatch(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>, %shape: tensor<2xi64>,
    %init: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // expected-error@+1 {{input and output element types must match}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<2xi64>)
      outs(%init : tensor<2x3xf32>) : tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// -----

func.func @expand_output_rank(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>, %shape: tensor<4xi64>,
    %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{output rank must equal max(input rank, shape length); expected 4, got 2}}
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<2x3xf16>, tensor<4xi64>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

func.func @expand_device_shape(
    %ctx: !hipsr.context,
    %input: memref<2x3xf16, #hipsr.mem<device>>,
    %shape: memref<2xi64, #hipsr.mem<device>>,
    %init: memref<2x3xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{operand #2 must be ranked tensor or host memref}}
  hipsr.expand(%ctx)
      ins(%input, %shape : memref<2x3xf16, #hipsr.mem<device>>,
                            memref<2xi64, #hipsr.mem<device>>)
      outs(%init : memref<2x3xf16, #hipsr.mem<device>>)
  return
}
