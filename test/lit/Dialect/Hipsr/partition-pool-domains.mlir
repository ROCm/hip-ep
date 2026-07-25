// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file -hipsr-partition-pool-domains %s | FileCheck %s

// A DPS chain gets fresh placeholders inside its assigned domain.
// CHECK-LABEL: func.func @cast_matmul_chain
// CHECK-SAME: (%[[CTX:.*]]: !hipsr.context, %[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x2xf16>)
// CHECK-NOT: hipsr.placeholder
// CHECK: hipsr.pool_domain(%[[CTX]], %[[A]], %[[B]] : !hipsr.context, tensor<4x8xf32>, tensor<8x2xf16>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DA:.*]]: tensor<4x8xf32>, %[[DB:.*]]: tensor<8x2xf16>):
// CHECK: %[[CAST_INIT:.*]] = hipsr.placeholder : tensor<4x8xf16>
// CHECK: %[[CAST:.*]] = hipsr.cast(%[[DCTX]]) ins(%[[DA]] : tensor<4x8xf32>) outs(%[[CAST_INIT]] : tensor<4x8xf16>)
// CHECK: %[[MM_INIT:.*]] = hipsr.placeholder : tensor<4x2xf16>
// CHECK: %[[MM:.*]] = hipsr.matmul(%[[DCTX]]) ins(%[[CAST]], %[[DB]]
// CHECK-SAME: outs(%[[MM_INIT]] : tensor<4x2xf16>)
// CHECK: hipsr.pool_domain_yield %[[MM]] : tensor<4x2xf16>
// CHECK: } -> tensor<4x2xf16>

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

// A placeholder is recreated immediately before its sole DPS consumer.
// CHECK-LABEL: func.func @dps_init_producer
// CHECK-SAME: (%[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4x8xf32>)
// CHECK-NOT: hipsr.placeholder
// CHECK: %[[DOMAIN:.*]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4x8xf32>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.*]]: !hipsr.context, %[[DINPUT:.*]]: tensor<4x8xf32>):
// CHECK-NEXT: %[[INIT:.*]] = hipsr.placeholder : tensor<4x8xf16>
// CHECK-NEXT: %[[CAST:.*]] = hipsr.cast(%[[DCTX]]) ins(%[[DINPUT]] : tensor<4x8xf32>) outs(%[[INIT]] : tensor<4x8xf16>)
// CHECK-NEXT: hipsr.pool_domain_yield %[[CAST]] : tensor<4x8xf16>
// CHECK: return %[[DOMAIN]] : tensor<4x8xf16>

func.func @dps_init_producer(%ctx: !hipsr.context,
                             %input: tensor<4x8xf32>)
    -> tensor<4x8xf16> {
  %init = hipsr.placeholder : tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Planning ignores an early placeholder and recreates it by its later consumer.
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

// Escaping values keep deterministic result and replacement order.
// CHECK-LABEL: func.func @multiple_results
// CHECK: %[[DOMAIN:.*]]:2 = hipsr.pool_domain(
// CHECK: %[[SUM:.*]] = arith.addf
// CHECK: %[[PRODUCT:.*]] = arith.mulf
// CHECK: hipsr.pool_domain_yield %[[SUM]], %[[PRODUCT]] : f32, f32
// CHECK: return %[[DOMAIN]]#1, %[[DOMAIN]]#0 : f32, f32

func.func @multiple_results(%lhs: f32, %rhs: f32) -> (f32, f32) {
  %sum = arith.addf %lhs, %rhs : f32
  %product = arith.mulf %lhs, %rhs : f32
  return %product, %sum : f32, f32
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
