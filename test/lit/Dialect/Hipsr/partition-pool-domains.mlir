// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// CSE before or after partitioning must not merge placeholders used by
// different ops.
// Every pool domain receives the function's first hipsr context argument.
// RUN: hip-mlir-opt --split-input-file -cse -hipsr-partition-pool-domains %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file -hipsr-partition-pool-domains -cse %s | FileCheck %s

// Expand with a shape attribute stays in its domain. Runtime Expand starts the
// next domain.
// CHECK-LABEL: func.func @expand_barrier_modes(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<?x4xf32>,
// CHECK-SAME: %[[SHAPE:.*]]: tensor<2xi64>)
// CHECK-SAME: -> tensor<?x4xf16> {
// CHECK-NEXT: %[[FIRST_DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<?x4xf32>) {
// CHECK-NEXT: ^bb0(%[[FIRST_CTX:.*]]: !hipsr.context, %[[FIRST_INPUT:.*]]: tensor<?x4xf32>):
// CHECK-NEXT: %[[CAST_INIT:.*]] = hipsr.placeholder : tensor<?x4xf16>
// CHECK-NEXT: %[[CAST:.*]] = hipsr.cast(%[[FIRST_CTX]]) ins(%[[FIRST_INPUT]] : tensor<?x4xf32>) outs(%[[CAST_INIT]] : tensor<?x4xf16>) : tensor<?x4xf16>
// CHECK-NEXT: %[[SHAPE_ATTR_INIT:.*]] = hipsr.placeholder : tensor<?x4xf16>
// CHECK-NEXT: %[[SHAPE_ATTR_EXPANDED:.*]] = hipsr.expand(%[[FIRST_CTX]]) ins(%[[CAST]] : tensor<?x4xf16>) outs(%[[SHAPE_ATTR_INIT]] : tensor<?x4xf16>) {shape_attr = array<i64: 1, 4>} : tensor<?x4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[SHAPE_ATTR_EXPANDED]] : tensor<?x4xf16>
// CHECK-NEXT: } -> tensor<?x4xf16>
// CHECK-NEXT: %[[EXPAND_DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[FIRST_DOMAIN]], %[[SHAPE]] : !hipsr.context, tensor<?x4xf16>, tensor<2xi64>) {
// CHECK-NEXT: ^bb0(%[[EXPAND_CTX:.*]]: !hipsr.context, %[[EXPAND_INPUT:.*]]: tensor<?x4xf16>, %[[EXPAND_SHAPE:.*]]: tensor<2xi64>):
// CHECK-NEXT: %[[EXPAND_INIT:.*]] = hipsr.placeholder : tensor<?x4xf16>
// CHECK-NEXT: %[[EXPANDED:.*]] = hipsr.expand(%[[EXPAND_CTX]]) ins(%[[EXPAND_INPUT]], %[[EXPAND_SHAPE]] : tensor<?x4xf16>, tensor<2xi64>) outs(%[[EXPAND_INIT]] : tensor<?x4xf16>) : tensor<?x4xf16>
// CHECK-NEXT: %[[ADD_INIT:.*]] = hipsr.placeholder : tensor<?x4xf16>
// CHECK-NEXT: %[[RESULT:.*]] = hipsr.add(%[[EXPAND_CTX]]) ins(%[[EXPANDED]], %[[EXPANDED]] : tensor<?x4xf16>, tensor<?x4xf16>) outs(%[[ADD_INIT]] : tensor<?x4xf16>) : tensor<?x4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[RESULT]] : tensor<?x4xf16>
// CHECK-NEXT: } -> tensor<?x4xf16>
// CHECK-NEXT: return %[[EXPAND_DOMAIN]] : tensor<?x4xf16>
// CHECK-NEXT: }
func.func @expand_barrier_modes(
    %ctx: !hipsr.context, %input: tensor<?x4xf32>, %shape: tensor<2xi64>)
    -> tensor<?x4xf16> {
  %cast_init = hipsr.placeholder : tensor<?x4xf16>
  %cast = hipsr.cast(%ctx) ins(%input : tensor<?x4xf32>)
      outs(%cast_init : tensor<?x4xf16>) : tensor<?x4xf16>
  %shape_attr_init = hipsr.placeholder : tensor<?x4xf16>
  %shape_attr_expanded = hipsr.expand(%ctx)
      ins(%cast : tensor<?x4xf16>)
      outs(%shape_attr_init : tensor<?x4xf16>)
      {shape_attr = array<i64: 1, 4>} : tensor<?x4xf16>
  %runtime_init = hipsr.placeholder : tensor<?x4xf16>
  %runtime_expanded = hipsr.expand(%ctx)
      ins(%shape_attr_expanded, %shape : tensor<?x4xf16>, tensor<2xi64>)
      outs(%runtime_init : tensor<?x4xf16>) : tensor<?x4xf16>
  %add_init = hipsr.placeholder : tensor<?x4xf16>
  %result = hipsr.add(%ctx)
      ins(%runtime_expanded, %runtime_expanded
          : tensor<?x4xf16>, tensor<?x4xf16>)
      outs(%add_init : tensor<?x4xf16>) : tensor<?x4xf16>
  return %result : tensor<?x4xf16>
}

// -----

// Nested regions use domain arguments. The early placeholder moves before
// Cast, while other operands and results keep operation order.
// CHECK-LABEL: func.func @nested_shape_region(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<?x8xf32>,
// CHECK-SAME: %[[CAPTURE:.*]]: i32) -> (tensor<?x8xf16>, i32) {
// CHECK-NEXT: %[[DOMAIN:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[CAPTURE]], %[[INPUT]] : !hipsr.context, i32, tensor<?x8xf32>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DCAPTURE:.*]]: i32, %[[DINPUT:.*]]: tensor<?x8xf32>):
// CHECK-NEXT: %[[REGION:.*]] = scf.execute_region -> i32 {
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[DCAPTURE]], %[[DCAPTURE]] : i32
// CHECK-NEXT: scf.yield %[[SUM]] : i32
// CHECK-NEXT: }
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
// CHECK-NEXT: hipsr.pool_domain_yield %[[REGION]], %[[CAST]] : i32, tensor<?x8xf16>
// CHECK-NEXT: } -> i32, tensor<?x8xf16>
// CHECK-NEXT: return %[[DOMAIN]]#1, %[[DOMAIN]]#0 : tensor<?x8xf16>, i32
// CHECK-NEXT: }
func.func @nested_shape_region(%ctx: !hipsr.context,
                               %input: tensor<?x8xf32>,
                               %capture: i32)
    -> (tensor<?x8xf16>, i32) {
  %init = hipsr.placeholder : tensor<?x8xf16>
  %region = scf.execute_region -> i32 {
    %sum = arith.addi %capture, %capture : i32
    scf.yield %sum : i32
  }
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
func.func @zero_result_retained(
    %ctx: !hipsr.context, %buffer: memref<4xi32>, %value: i32, %index: index) {
  memref.store %value, %buffer[%index] : memref<4xi32>
  return
}

// Empty functions and declarations are unchanged.
// CHECK-LABEL: func.func @empty()
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @empty() {
  return
}

// CHECK-LABEL: func.func private @declaration(i32) -> i32
// CHECK-NEXT: }
func.func private @declaration(i32) -> i32
