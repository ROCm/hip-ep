// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// CSE before or after partitioning must not merge placeholders used by
// different ops.
// Every pool domain receives the function's first hipsr context argument.
// RUN: hip-mlir-opt --split-input-file -cse -hipsr-partition-pool-domains %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file -hipsr-partition-pool-domains -cse %s | FileCheck %s

// Runtime Expand starts a domain.
// CHECK-LABEL: func.func @runtime_expand_splits_domains(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<?x4xf32>,
// CHECK-SAME: %[[SHAPE:.*]]: tensor<2xi64>)
// CHECK-SAME: -> tensor<?x4xf16> {
// CHECK-NEXT: %[[CAST_DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<?x4xf32>) {
// CHECK-NEXT: ^bb0(%[[CAST_CTX:.*]]: !hipsr.context, %[[CAST_INPUT:.*]]: tensor<?x4xf32>):
// CHECK-NEXT: %[[CAST_INIT:.*]] = hipsr.placeholder : tensor<?x4xf16>
// CHECK-NEXT: %[[CAST:.*]] = hipsr.cast(%[[CAST_CTX]]) ins(%[[CAST_INPUT]] : tensor<?x4xf32>) outs(%[[CAST_INIT]] : tensor<?x4xf16>) : tensor<?x4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[CAST]] : tensor<?x4xf16>
// CHECK-NEXT: } -> tensor<?x4xf16>
// CHECK-NEXT: %[[EXPAND_DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[CAST_DOMAIN]], %[[SHAPE]] : !hipsr.context, tensor<?x4xf16>, tensor<2xi64>) {
// CHECK-NEXT: ^bb0(%[[EXPAND_CTX:.*]]: !hipsr.context, %[[EXPAND_INPUT:.*]]: tensor<?x4xf16>, %[[EXPAND_SHAPE:.*]]: tensor<2xi64>):
// CHECK-NEXT: %[[EXPAND_INIT:.*]] = hipsr.placeholder : tensor<?x4xf16>
// CHECK-NEXT: %[[EXPANDED:.*]] = hipsr.expand(%[[EXPAND_CTX]]) ins(%[[EXPAND_INPUT]], %[[EXPAND_SHAPE]] : tensor<?x4xf16>, tensor<2xi64>) outs(%[[EXPAND_INIT]] : tensor<?x4xf16>) : tensor<?x4xf16>
// CHECK-NEXT: %[[ADD_INIT:.*]] = hipsr.placeholder : tensor<?x4xf16>
// CHECK-NEXT: %[[RESULT:.*]] = hipsr.add(%[[EXPAND_CTX]]) ins(%[[EXPANDED]], %[[EXPANDED]] : tensor<?x4xf16>, tensor<?x4xf16>) outs(%[[ADD_INIT]] : tensor<?x4xf16>) : tensor<?x4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[RESULT]] : tensor<?x4xf16>
// CHECK-NEXT: } -> tensor<?x4xf16>
// CHECK-NEXT: return %[[EXPAND_DOMAIN]] : tensor<?x4xf16>
// CHECK-NEXT: }
func.func @runtime_expand_splits_domains(
    %ctx: !hipsr.context, %input: tensor<?x4xf32>, %shape: tensor<2xi64>)
    -> tensor<?x4xf16> {
  %cast_init = hipsr.placeholder : tensor<?x4xf16>
  %cast = hipsr.cast(%ctx) ins(%input : tensor<?x4xf32>)
      outs(%cast_init : tensor<?x4xf16>) : tensor<?x4xf16>
  %expand_init = hipsr.placeholder : tensor<?x4xf16>
  %expanded = hipsr.expand(%ctx)
      ins(%cast, %shape : tensor<?x4xf16>, tensor<2xi64>)
      outs(%expand_init : tensor<?x4xf16>) : tensor<?x4xf16>
  %add_init = hipsr.placeholder : tensor<?x4xf16>
  %result = hipsr.add(%ctx)
      ins(%expanded, %expanded : tensor<?x4xf16>, tensor<?x4xf16>)
      outs(%add_init : tensor<?x4xf16>) : tensor<?x4xf16>
  return %result : tensor<?x4xf16>
}

// -----

// With a shape attribute, Expand needs no runtime read and stays in the current
// domain even with a dynamic result.
// CHECK-LABEL: func.func @shape_attr_dynamic_result_stays_in_domain(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<?x3xf32>)
// CHECK-SAME: -> tensor<?x3xf16> {
// CHECK-NEXT: %[[DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<?x3xf32>) {
// CHECK-NEXT: ^bb0(%[[DOMAIN_CTX:.*]]: !hipsr.context, %[[DOMAIN_INPUT:.*]]: tensor<?x3xf32>):
// CHECK-NEXT: %[[CAST_INIT:.*]] = hipsr.placeholder : tensor<?x3xf16>
// CHECK-NEXT: %[[CAST:.*]] = hipsr.cast(%[[DOMAIN_CTX]]) ins(%[[DOMAIN_INPUT]] : tensor<?x3xf32>) outs(%[[CAST_INIT]] : tensor<?x3xf16>) : tensor<?x3xf16>
// CHECK-NEXT: %[[EXPAND_INIT:.*]] = hipsr.placeholder : tensor<?x3xf16>
// CHECK-NEXT: %[[RESULT:.*]] = hipsr.expand(%[[DOMAIN_CTX]]) ins(%[[CAST]] : tensor<?x3xf16>) outs(%[[EXPAND_INIT]] : tensor<?x3xf16>) {shape_attr = array<i64: 1, 3>} : tensor<?x3xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[RESULT]] : tensor<?x3xf16>
// CHECK-NEXT: } -> tensor<?x3xf16>
// CHECK-NEXT: return %[[DOMAIN]] : tensor<?x3xf16>
// CHECK-NEXT: }
func.func @shape_attr_dynamic_result_stays_in_domain(
    %ctx: !hipsr.context, %input: tensor<?x3xf32>) -> tensor<?x3xf16> {
  %cast_init = hipsr.placeholder : tensor<?x3xf16>
  %cast = hipsr.cast(%ctx) ins(%input : tensor<?x3xf32>)
      outs(%cast_init : tensor<?x3xf16>) : tensor<?x3xf16>
  %expand_init = hipsr.placeholder : tensor<?x3xf16>
  %result = hipsr.expand(%ctx)
      ins(%cast : tensor<?x3xf16>)
      outs(%expand_init : tensor<?x3xf16>)
      {shape_attr = array<i64: 1, 3>} : tensor<?x3xf16>
  return %result : tensor<?x3xf16>
}

// -----

// A DPS chain and its placeholders share one domain.
// CHECK-LABEL: func.func @cast_matmul_chain(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x2xf16>)
// CHECK-SAME: -> tensor<4x2xf16> {
// CHECK-NEXT: %[[DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[A]], %[[B]] : !hipsr.context, tensor<4x8xf32>, tensor<8x2xf16>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DA:.*]]: tensor<4x8xf32>, %[[DB:.*]]: tensor<8x2xf16>):
// CHECK-NEXT: %[[CAST_INIT:.*]] = hipsr.placeholder : tensor<4x8xf16>
// CHECK-NEXT: %[[CAST:.*]] = hipsr.cast(%[[DCTX]]) ins(%[[DA]] : tensor<4x8xf32>) outs(%[[CAST_INIT]] : tensor<4x8xf16>) : tensor<4x8xf16>
// CHECK-NEXT: %[[MM_INIT:.*]] = hipsr.placeholder : tensor<4x2xf16>
// CHECK-NEXT: %[[MM:.*]] = hipsr.matmul(%[[DCTX]]) ins(%[[CAST]], %[[DB]]
// CHECK-SAME: outs(%[[MM_INIT]] : tensor<4x2xf16>) : tensor<4x2xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[MM]] : tensor<4x2xf16>
// CHECK-NEXT: } -> tensor<4x2xf16>
// CHECK-NEXT: return %[[DOMAIN]] : tensor<4x2xf16>
// CHECK-NEXT: }

func.func @cast_matmul_chain(
    %ctx: !hipsr.context, %a: tensor<4x8xf32>, %b: tensor<8x2xf16>)
    -> tensor<4x2xf16> {
  %cast_init = hipsr.placeholder : tensor<4x8xf16>
  %cast = hipsr.cast(%ctx) ins(%a : tensor<4x8xf32>)
                     outs(%cast_init : tensor<4x8xf16>) : tensor<4x8xf16>
  %matmul_init = hipsr.placeholder : tensor<4x2xf16>
  %result = hipsr.matmul(%ctx)
      ins(%cast, %b : tensor<4x8xf16>, tensor<8x2xf16>)
      outs(%matmul_init : tensor<4x2xf16>) : tensor<4x2xf16>
  return %result : tensor<4x2xf16>
}

// -----

// An early placeholder moves next to the op that uses it.
// CHECK-LABEL: func.func @deferred_dps_init(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4x8xf32>,
// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32) -> (tensor<4x8xf16>, f32) {
// CHECK-NEXT: %[[DOMAIN:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[LHS]], %[[RHS]], %[[INPUT]] : !hipsr.context, f32, f32, tensor<4x8xf32>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DLHS:.*]]: f32, %[[DRHS:.*]]: f32, %[[DINPUT:.*]]: tensor<4x8xf32>):
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[DLHS]], %[[DRHS]] : f32
// CHECK-NEXT: %[[INIT:.*]] = hipsr.placeholder : tensor<4x8xf16>
// CHECK-NEXT: %[[CAST:.*]] = hipsr.cast(%[[DCTX]]) ins(%[[DINPUT]] : tensor<4x8xf32>) outs(%[[INIT]] : tensor<4x8xf16>) : tensor<4x8xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[SUM]], %[[CAST]] : f32, tensor<4x8xf16>
// CHECK-NEXT: } -> f32, tensor<4x8xf16>
// CHECK-NEXT: return %[[DOMAIN]]#1, %[[DOMAIN]]#0 : tensor<4x8xf16>, f32
// CHECK-NEXT: }

func.func @deferred_dps_init(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>, %lhs: f32, %rhs: f32)
    -> (tensor<4x8xf16>, f32) {
  %init = hipsr.placeholder : tensor<4x8xf16>
  %sum = arith.addf %lhs, %rhs : f32
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result, %sum : tensor<4x8xf16>, f32
}

// -----

// Context is first. Other domain operands keep first-use order.
// CHECK-LABEL: func.func @operand_order(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[A:.*]]: i32, %[[B:.*]]: i32,
// CHECK-SAME: %[[C:.*]]: i32) -> i32 {
// CHECK-NEXT: %[[DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[C]], %[[A]], %[[B]] : !hipsr.context, i32, i32, i32) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DC:.*]]: i32, %[[DA:.*]]: i32, %[[DB:.*]]: i32):
// CHECK-NEXT: %[[FIRST:.*]] = arith.addi %[[DC]], %[[DA]] : i32
// CHECK-NEXT: %[[SECOND:.*]] = arith.addi %[[FIRST]], %[[DB]] : i32
// CHECK-NEXT: hipsr.pool_domain_yield %[[SECOND]] : i32
// CHECK-NEXT: } -> i32
// CHECK-NEXT: return %[[DOMAIN]] : i32
// CHECK-NEXT: }

func.func @operand_order(
    %ctx: !hipsr.context, %a: i32, %b: i32, %c: i32) -> i32 {
  %first = arith.addi %c, %a : i32
  %second = arith.addi %first, %b : i32
  return %second : i32
}

// -----

// Values used in a nested shape region become domain inputs.
// CHECK-LABEL: func.func @nested_shape_region(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<?x8xf32>) -> tensor<?x8xf16> {
// CHECK-NEXT: %[[DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<?x8xf32>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DINPUT:.*]]: tensor<?x8xf32>):
// CHECK-NEXT: %[[INIT:.*]] = hipsr.placeholder : tensor<?x8xf16>
// CHECK-NEXT: %[[CAST:.*]] = hipsr.cast(%[[DCTX]]) ins(%[[DINPUT]] : tensor<?x8xf32>) outs(%[[INIT]] : tensor<?x8xf16>) : tensor<?x8xf16> shape_region {
// CHECK-NEXT: ^bb0(%[[SHAPE_INPUT:.*]]: tensor<?x8xf32>):
// CHECK-NEXT: %[[SHAPE:.*]] = shape.shape_of %[[SHAPE_INPUT]] : tensor<?x8xf32> -> tensor<2xindex>
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[D0:.*]] = shape.get_extent %[[SHAPE]], %[[C0]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[D1:.*]] = shape.get_extent %[[SHAPE]], %[[C1]] : tensor<2xindex>, index -> index
// CHECK-NEXT: hipsr.shape_yield (%[[D0]], %[[D1]]) : [f16]
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.pool_domain_yield %[[CAST]] : tensor<?x8xf16>
// CHECK-NEXT: } -> tensor<?x8xf16>
// CHECK-NEXT: return %[[DOMAIN]] : tensor<?x8xf16>
// CHECK-NEXT: }

func.func @nested_shape_region(%ctx: !hipsr.context,
                               %input: tensor<?x8xf32>)
    -> tensor<?x8xf16> {
  %init = hipsr.placeholder : tensor<?x8xf16>
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16> shape_region {
  ^bb0(%shape_input: tensor<?x8xf32>):
    %shape = shape.shape_of %shape_input : tensor<?x8xf32> -> tensor<2xindex>
    %c0 = arith.constant 0 : index
    %d0 = shape.get_extent %shape, %c0 : tensor<2xindex>, index -> index
    %c1 = arith.constant 1 : index
    %d1 = shape.get_extent %shape, %c1 : tensor<2xindex>, index -> index
    hipsr.shape_yield (%d0, %d1) : [f16]
  }
  return %0 : tensor<?x8xf16>
}

// -----

// Nested regions use domain arguments after the split.
// CHECK-LABEL: func.func @implicit_region_capture(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[CAPTURE:.*]]: i32) -> i32 {
// CHECK-NEXT: %[[DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[CAPTURE]] : !hipsr.context, i32) {
// CHECK-NEXT: ^bb0(%[[DOMAIN_CTX:.*]]: !hipsr.context, %[[DOMAIN_CAPTURE:.*]]: i32):
// CHECK-NEXT: %[[REGION:.*]] = scf.execute_region -> i32 {
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[DOMAIN_CAPTURE]], %[[DOMAIN_CAPTURE]] : i32
// CHECK-NEXT: scf.yield %[[SUM]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.pool_domain_yield %[[REGION]] : i32
// CHECK-NEXT: } -> i32
// CHECK-NEXT: return %[[DOMAIN]] : i32
// CHECK-NEXT: }

func.func @implicit_region_capture(
    %ctx: !hipsr.context, %capture: i32) -> i32 {
  %0 = scf.execute_region -> i32 {
    %1 = arith.addi %capture, %capture : i32
    scf.yield %1 : i32
  }
  return %0 : i32
}

// -----

// Side-effecting ops with no results stay in their domain.
// CHECK-LABEL: func.func @zero_result_retained(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[BUFFER:.*]]: memref<4xi32>,
// CHECK-SAME: %[[VALUE:.*]]: i32, %[[INDEX:.*]]: index) {
// CHECK-NEXT: hipsr.pool_domain(%[[CTX]], %[[VALUE]], %[[BUFFER]], %[[INDEX]] : !hipsr.context, i32, memref<4xi32>, index) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DVALUE:.*]]: i32, %[[DBUFFER:.*]]: memref<4xi32>, %[[DINDEX:.*]]: index):
// CHECK-NEXT: memref.store %[[DVALUE]], %[[DBUFFER]]{{\[}}%[[DINDEX]]] : memref<4xi32>
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }

func.func @zero_result_retained(
    %ctx: !hipsr.context, %buffer: memref<4xi32>, %value: i32, %index: index) {
  memref.store %value, %buffer[%index] : memref<4xi32>
  return
}

// -----

// Empty functions are unchanged.
// CHECK-LABEL: func.func @empty()
// CHECK-NEXT: return
// CHECK-NEXT: }

func.func @empty() {
  return
}

// -----

// Declarations are unchanged.
// CHECK-LABEL: func.func private @declaration(i32) -> i32
// CHECK-NEXT: }

func.func private @declaration(i32) -> i32
