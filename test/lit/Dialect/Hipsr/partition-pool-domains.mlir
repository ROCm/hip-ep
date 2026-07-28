// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// CSE before or after partitioning must preserve consumer-specific placeholders.
// RUN: hip-mlir-opt --split-input-file -cse -hipsr-partition-pool-domains %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file -hipsr-partition-pool-domains -cse %s | FileCheck %s

// A runtime Expand starts a second domain. Same-typed placeholders remain
// distinct across CSE and move beside their Cast, Expand, and Add consumers.
// CHECK-LABEL: func.func @runtime_expand_splits_domains(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<?x4xf32>,
// CHECK-SAME: %[[SHAPE:.*]]: tensor<2xi64>)
// CHECK: %[[CAST_DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<?x4xf32>) {
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

// A shape_attr avoids a runtime shape-data read, so Expand stays in Cast's
// domain even though broadcasting preserves a dynamic result dimension.
// CHECK-LABEL: func.func @shape_attr_dynamic_result_stays_in_domain(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<?x3xf32>)
// CHECK: %[[DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<?x3xf32>) {
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

// A DPS chain and its placeholders move into one domain.
// CHECK-LABEL: func.func @cast_matmul_chain
// CHECK-SAME: (%[[CTX:.*]]: !hipsr.context, %[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x2xf16>)
// CHECK-NOT: hipsr.placeholder
// CHECK: hipsr.pool_domain(%[[CTX]], %[[A]], %[[B]] : !hipsr.context, tensor<4x8xf32>, tensor<8x2xf16>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DA:.*]]: tensor<4x8xf32>, %[[DB:.*]]: tensor<8x2xf16>):
// CHECK-NEXT: %[[CAST_INIT:.*]] = hipsr.placeholder : tensor<4x8xf16>
// CHECK-NEXT: %[[CAST:.*]] = hipsr.cast(%[[DCTX]]) ins(%[[DA]] : tensor<4x8xf32>) outs(%[[CAST_INIT]] : tensor<4x8xf16>)
// CHECK-NEXT: %[[MM_INIT:.*]] = hipsr.placeholder : tensor<4x2xf16>
// CHECK-NEXT: %[[MM:.*]] = hipsr.matmul(%[[DCTX]]) ins(%[[CAST]], %[[DB]]
// CHECK-SAME: outs(%[[MM_INIT]] : tensor<4x2xf16>)
// CHECK-NEXT: hipsr.pool_domain_yield %[[MM]] : tensor<4x2xf16>
// CHECK-NEXT: } -> tensor<4x2xf16>

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

// Planning ignores an early placeholder and moves it beside its consumer.
// CHECK-LABEL: func.func @deferred_dps_init
// CHECK-NOT: hipsr.placeholder
// CHECK: %[[DOMAIN:.*]]:2 = hipsr.pool_domain(
// CHECK: %[[SUM:.*]] = arith.addf
// CHECK-NEXT: %[[INIT:.*]] = hipsr.placeholder : tensor<4x8xf16>
// CHECK-NEXT: %[[CAST:.*]] = hipsr.cast
// CHECK-SAME: outs(%[[INIT]] : tensor<4x8xf16>)
// CHECK: hipsr.pool_domain_yield %[[SUM]], %[[CAST]]
// CHECK: return %[[DOMAIN]]#1, %[[DOMAIN]]#0

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

// Captured operands keep their first-use order.
// CHECK-LABEL: func.func @operand_order
// CHECK: hipsr.pool_domain(%[[C:.*]], %[[A:.*]], %[[B:.*]] : i32, i32, i32) {
// CHECK-NEXT: ^bb0(%[[DC:.*]]: i32, %[[DA:.*]]: i32, %[[DB:.*]]: i32):
// CHECK: %[[FIRST:.*]] = arith.addi %[[DC]], %[[DA]] : i32
// CHECK: %[[SECOND:.*]] = arith.addi %[[FIRST]], %[[DB]] : i32
// CHECK: hipsr.pool_domain_yield %[[SECOND]] : i32

func.func @operand_order(%a: i32, %b: i32, %c: i32) -> i32 {
  %first = arith.addi %c, %a : i32
  %second = arith.addi %first, %b : i32
  return %second : i32
}

// -----

// Values used by a nested shape region become explicit domain inputs.
// CHECK-LABEL: func.func @nested_shape_region
// CHECK: hipsr.pool_domain(%[[CTX:.*]], %[[INPUT:.*]]
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DINPUT:.*]]: tensor<?x8xf32>):
// CHECK: %[[INIT:.*]] = hipsr.placeholder : tensor<?x8xf16>
// CHECK: hipsr.cast(%[[DCTX]]) ins(%[[DINPUT]] : tensor<?x8xf32>) outs(%[[INIT]] : tensor<?x8xf16>)
// CHECK: shape_region {
// CHECK-NEXT: ^bb0(%[[SHAPE_INPUT:.*]]: tensor<?x8xf32>):
// CHECK: shape.shape_of %[[SHAPE_INPUT]]
// CHECK: hipsr.shape_yield

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

// Implicit captures in descendant regions are remapped to domain arguments.
// CHECK-LABEL: func.func @implicit_region_capture(
// CHECK-SAME: %[[CAPTURE:.*]]: i32)
// CHECK: hipsr.pool_domain(%[[CAPTURE]] : i32) {
// CHECK-NEXT: ^bb0(%[[DOMAIN_CAPTURE:.*]]: i32):
// CHECK: scf.execute_region -> i32 {
// CHECK: arith.addi %[[DOMAIN_CAPTURE]], %[[DOMAIN_CAPTURE]] : i32

func.func @implicit_region_capture(%capture: i32) -> i32 {
  %0 = scf.execute_region -> i32 {
    %1 = arith.addi %capture, %capture : i32
    scf.yield %1 : i32
  }
  return %0 : i32
}

// -----

// A side-effecting operation with no result remains in its domain.
// CHECK-LABEL: func.func @zero_result_retained
// CHECK: hipsr.pool_domain(
// CHECK: memref.store
// CHECK-NEXT: }

func.func @zero_result_retained(%buffer: memref<4xi32>, %value: i32,
                                %index: index) {
  memref.store %value, %buffer[%index] : memref<4xi32>
  return
}

// -----

// An empty function is unchanged.
// CHECK-LABEL: func.func @empty()
// CHECK-NEXT: return
// CHECK-NEXT: }

func.func @empty() {
  return
}

// -----

// A declaration is unchanged.
// CHECK-LABEL: func.func private @declaration
// CHECK-NOT: hipsr.pool_domain
// CHECK-NEXT: }

func.func private @declaration(i32) -> i32
