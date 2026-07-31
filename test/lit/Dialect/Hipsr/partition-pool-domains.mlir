// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Every pool domain receives the function's first hipsr context argument.
// RUN: hip-mlir-opt --split-input-file -hipsr-partition-pool-domains %s | FileCheck %s

// Placeholder categories do not split domains. This fixture uses function
// arguments as shape-graph roots.
// CHECK-LABEL: func.func @expand_barrier_modes(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x4xf32>,
// CHECK-SAME: %[[SHAPE:.+]]: tensor<2xi64>)
// CHECK-SAME: -> tensor<?x4xf16> {
// CHECK-NEXT: %[[DOMAIN:.+]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]], %[[SHAPE]] : !hipsr.context, tensor<?x4xf32>, tensor<2xi64>) {
// CHECK-NEXT: ^bb0(%[[DOMAIN_CTX:.+]]: !hipsr.context, %[[SHAPE_ROOT:.+]]: tensor<?x4xf32>, %[[DOMAIN_SHAPE:.+]]: tensor<2xi64>):
// CHECK-NEXT: %[[CAST_INIT:.+]] = hipsr.placeholder(%[[DOMAIN_CTX]]) ins(%[[SHAPE_ROOT]] : tensor<?x4xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
// CHECK-NEXT: %[[CAST:.+]] = hipsr.cast(%[[DOMAIN_CTX]]) ins(%[[SHAPE_ROOT]] : tensor<?x4xf32>) outs(%[[CAST_INIT]] : tensor<?x4xf16>) : tensor<?x4xf16>
// CHECK-NEXT: %[[SHAPE_ATTR_INIT:.+]] = hipsr.placeholder(%[[DOMAIN_CTX]]) ins(%[[SHAPE_ROOT]] : tensor<?x4xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
// CHECK-NEXT: %[[SHAPE_ATTR_EXPANDED:.+]] = hipsr.expand(%[[DOMAIN_CTX]]) ins(%[[CAST]] : tensor<?x4xf16>) outs(%[[SHAPE_ATTR_INIT]] : tensor<?x4xf16>) {shape_attr = array<i64: 1, 4>} : tensor<?x4xf16>
// CHECK-NEXT: %[[EXPAND_INIT:.+]] = hipsr.placeholder(%[[DOMAIN_CTX]]) ins(%[[SHAPE_ROOT]], %[[DOMAIN_SHAPE]] : tensor<?x4xf32>, tensor<2xi64>) {type = #hipsr.placeholder_type<barrier>} : tensor<?x4xf16>
// CHECK-NEXT: %[[EXPANDED:.+]] = hipsr.expand(%[[DOMAIN_CTX]]) ins(%[[SHAPE_ATTR_EXPANDED]], %[[DOMAIN_SHAPE]] : tensor<?x4xf16>, tensor<2xi64>) outs(%[[EXPAND_INIT]] : tensor<?x4xf16>) : tensor<?x4xf16>
// CHECK-NEXT: %[[ADD_INIT:.+]] = hipsr.placeholder(%[[DOMAIN_CTX]]) ins(%[[SHAPE_ROOT]], %[[SHAPE_ROOT]] : tensor<?x4xf32>, tensor<?x4xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.add(%[[DOMAIN_CTX]]) ins(%[[EXPANDED]], %[[EXPANDED]] : tensor<?x4xf16>, tensor<?x4xf16>) outs(%[[ADD_INIT]] : tensor<?x4xf16>) : tensor<?x4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[RESULT]] : tensor<?x4xf16>
// CHECK-NEXT: } -> tensor<?x4xf16>
// CHECK-NEXT: return %[[DOMAIN]] : tensor<?x4xf16>
// CHECK-NEXT: }
func.func @expand_barrier_modes(
    %ctx: !hipsr.context, %input: tensor<?x4xf32>, %shape: tensor<2xi64>)
    -> tensor<?x4xf16> {
  %cast_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x4xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
  %cast = hipsr.cast(%ctx) ins(%input : tensor<?x4xf32>)
      outs(%cast_init : tensor<?x4xf16>) : tensor<?x4xf16>
  %shape_attr_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x4xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
  %shape_attr_expanded = hipsr.expand(%ctx)
      ins(%cast : tensor<?x4xf16>)
      outs(%shape_attr_init : tensor<?x4xf16>)
      {shape_attr = array<i64: 1, 4>} : tensor<?x4xf16>
  %runtime_init = hipsr.placeholder(%ctx)
      ins(%input, %shape : tensor<?x4xf32>, tensor<2xi64>)
      {type = #hipsr.placeholder_type<barrier>} : tensor<?x4xf16>
  %runtime_expanded = hipsr.expand(%ctx)
      ins(%shape_attr_expanded, %shape : tensor<?x4xf16>, tensor<2xi64>)
      outs(%runtime_init : tensor<?x4xf16>) : tensor<?x4xf16>
  %add_init = hipsr.placeholder(%ctx)
      ins(%input, %input : tensor<?x4xf32>, tensor<?x4xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
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
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x8xf32>,
// CHECK-SAME: %[[CAPTURE:.+]]: i32) -> (tensor<?x8xf16>, i32) {
// CHECK-NEXT: %[[DOMAIN:.+]]:2 = hipsr.pool_domain(%[[CTX]], %[[CAPTURE]], %[[INPUT]] : !hipsr.context, i32, tensor<?x8xf32>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DCAPTURE:.+]]: i32, %[[DINPUT:.+]]: tensor<?x8xf32>):
// CHECK-NEXT: %[[REGION:.+]] = scf.execute_region -> i32 {
// CHECK-NEXT: %[[SUM:.+]] = arith.addi %[[DCAPTURE]], %[[DCAPTURE]] : i32
// CHECK-NEXT: scf.yield %[[SUM]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[DCTX]]) ins(%[[DINPUT]] : tensor<?x8xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16> shape_region {
// CHECK-NEXT: ^bb0(%[[SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[CAST:.+]] = hipsr.cast(%[[DCTX]]) ins(%[[DINPUT]] : tensor<?x8xf32>) outs(%[[INIT]] : tensor<?x8xf16>) : tensor<?x8xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[REGION]], %[[CAST]] : i32, tensor<?x8xf16>
// CHECK-NEXT: } -> i32, tensor<?x8xf16>
// CHECK-NEXT: return %[[DOMAIN]]#1, %[[DOMAIN]]#0 : tensor<?x8xf16>, i32
// CHECK-NEXT: }
func.func @nested_shape_region(%ctx: !hipsr.context,
                               %input: tensor<?x8xf32>,
                               %capture: i32)
    -> (tensor<?x8xf16>, i32) {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
      shape_region {
  ^bb0(%input_shape: !shape.shape):
    hipsr.shape_yield %input_shape : !shape.shape
  }
  %region = scf.execute_region -> i32 {
    %sum = arith.addi %capture, %capture : i32
    scf.yield %sum : i32
  }
  %cast = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %cast, %region : tensor<?x8xf16>, i32
}

// -----

// A side-effecting op creates a domain with no results.
// CHECK-LABEL: func.func @zero_result_retained(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[BUFFER:.+]]: memref<4xi32>,
// CHECK-SAME: %[[VALUE:.+]]: i32, %[[INDEX:.+]]: index) {
// CHECK-NEXT: hipsr.pool_domain(%[[CTX]], %[[VALUE]], %[[BUFFER]], %[[INDEX]] : !hipsr.context, i32, memref<4xi32>, index) {
// CHECK-NEXT: ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DVALUE:.+]]: i32, %[[DBUFFER:.+]]: memref<4xi32>, %[[DINDEX:.+]]: index):
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
