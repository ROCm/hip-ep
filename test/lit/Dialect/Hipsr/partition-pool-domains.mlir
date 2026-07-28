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

// Context stays first and other operands keep first-use order. The early
// placeholder moves before Cast.
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

// Shape and SCF regions use the new domain arguments.
// CHECK-LABEL: func.func @nested_shape_region(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<?x8xf32>,
// CHECK-SAME: %[[CAPTURE:.*]]: i32) -> (tensor<?x8xf16>, i32) {
// CHECK-NEXT: %[[DOMAIN:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[INPUT]], %[[CAPTURE]] : !hipsr.context, tensor<?x8xf32>, i32) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DINPUT:.*]]: tensor<?x8xf32>, %[[DCAPTURE:.*]]: i32):
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
// CHECK-NEXT: %[[REGION:.*]] = scf.execute_region -> i32 {
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[DCAPTURE]], %[[DCAPTURE]] : i32
// CHECK-NEXT: scf.yield %[[SUM]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.pool_domain_yield %[[CAST]], %[[REGION]] : tensor<?x8xf16>, i32
// CHECK-NEXT: } -> tensor<?x8xf16>, i32
// CHECK-NEXT: return %[[DOMAIN]]#0, %[[DOMAIN]]#1 : tensor<?x8xf16>, i32
// CHECK-NEXT: }

func.func @nested_shape_region(%ctx: !hipsr.context,
                               %input: tensor<?x8xf32>,
                               %capture: i32)
    -> (tensor<?x8xf16>, i32) {
  %init = hipsr.placeholder : tensor<?x8xf16>
  %cast = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16> shape_region {
  ^bb0(%shape_input: tensor<?x8xf32>):
    %shape = shape.shape_of %shape_input : tensor<?x8xf32> -> tensor<2xindex>
    %c0 = arith.constant 0 : index
    %d0 = shape.get_extent %shape, %c0 : tensor<2xindex>, index -> index
    %c1 = arith.constant 1 : index
    %d1 = shape.get_extent %shape, %c1 : tensor<2xindex>, index -> index
    hipsr.shape_yield (%d0, %d1) : [f16]
  }
  %region = scf.execute_region -> i32 {
    %sum = arith.addi %capture, %capture : i32
    scf.yield %sum : i32
  }
  return %cast, %region : tensor<?x8xf16>, i32
}

// -----

// A side-effecting op creates a domain with no results.
// CHECK-LABEL: func.func @zero_result_retained(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[BUFFER:.*]]: memref<4xi32>,
// CHECK-SAME: %[[VALUE:.*]]: i32, %[[INDEX:.*]]: index) {
// CHECK-NEXT: hipsr.pool_domain(%[[CTX]], %[[VALUE]], %[[BUFFER]], %[[INDEX]] : !hipsr.context, i32, memref<4xi32>, index) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DVALUE:.*]]: i32, %[[DBUFFER:.*]]: memref<4xi32>, %[[DINDEX:.*]]: index):
// CHECK-NEXT: memref.store %[[DVALUE]], %[[DBUFFER]]{{\[}}%[[DINDEX]]] : memref<4xi32>
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }
// Empty functions and declarations are unchanged.
// CHECK-LABEL: func.func @empty()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @declaration(i32) -> i32
// CHECK-NEXT: }

func.func @zero_result_retained(
    %ctx: !hipsr.context, %buffer: memref<4xi32>, %value: i32, %index: index) {
  memref.store %value, %buffer[%index] : memref<4xi32>
  return
}

func.func @empty() {
  return
}

func.func private @declaration(i32) -> i32
