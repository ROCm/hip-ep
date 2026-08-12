// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.equal round-trip and verifier rules.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// A rank-0 operand broadcasts against any shape, which is how a graph compares
// a whole tensor against one token id.
// CHECK-LABEL: func.func @equal_broadcast_scalar(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context, %[[LHS:.+]]: tensor<?x?xi64>,
// CHECK-SAME:    %[[RHS:.+]]: tensor<i64>,
// CHECK-SAME:    %[[INIT:.+]]: tensor<?x?xui8>) -> tensor<?x?xui8> {
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.equal(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<?x?xi64>, tensor<i64>) outs(%[[INIT]] : tensor<?x?xui8>) : tensor<?x?xui8>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?xui8>
func.func @equal_broadcast_scalar(
    %ctx: !hipsr.context, %lhs: tensor<?x?xi64>, %rhs: tensor<i64>,
    %init: tensor<?x?xui8>) -> tensor<?x?xui8> {
  %result = hipsr.equal(%ctx)
      ins(%lhs, %rhs : tensor<?x?xi64>, tensor<i64>)
      outs(%init : tensor<?x?xui8>) : tensor<?x?xui8>
  return %result : tensor<?x?xui8>
}

// -----

// A unit extent broadcasts against a longer one, and the result may be i1.
// CHECK-LABEL: func.func @equal_broadcast_ranks(
// CHECK:         hipsr.equal(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<4x1xf16>, tensor<3xf16>) outs(%{{.+}} : tensor<4x3xi1>) : tensor<4x3xi1>
func.func @equal_broadcast_ranks(
    %ctx: !hipsr.context, %lhs: tensor<4x1xf16>, %rhs: tensor<3xf16>,
    %init: tensor<4x3xi1>) -> tensor<4x3xi1> {
  %result = hipsr.equal(%ctx)
      ins(%lhs, %rhs : tensor<4x1xf16>, tensor<3xf16>)
      outs(%init : tensor<4x3xi1>) : tensor<4x3xi1>
  return %result : tensor<4x3xi1>
}

// -----

// Comparing values of different types has no defined meaning.
func.func @equal_operand_element_mismatch(
    %ctx: !hipsr.context, %lhs: tensor<2x3xf16>, %rhs: tensor<2x3xf32>,
    %init: tensor<2x3xui8>) -> tensor<2x3xui8> {
  // expected-error@+1 {{lhs and rhs element types must match}}
  %result = hipsr.equal(%ctx)
      ins(%lhs, %rhs : tensor<2x3xf16>, tensor<2x3xf32>)
      outs(%init : tensor<2x3xui8>) : tensor<2x3xui8>
  return %result : tensor<2x3xui8>
}

// -----

// The result is a mask, not a value of the operands' type.
func.func @equal_output_element_type(
    %ctx: !hipsr.context, %lhs: tensor<2x3xf16>, %rhs: tensor<2x3xf16>,
    %init: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{output element type must be i1 or an 8-bit integer}}
  %result = hipsr.equal(%ctx)
      ins(%lhs, %rhs : tensor<2x3xf16>, tensor<2x3xf16>)
      outs(%init : tensor<2x3xf16>) : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

// Aligned extents of 3 and 4 neither match nor contain a one.
func.func @equal_incompatible_operands(
    %ctx: !hipsr.context, %lhs: tensor<2x3xf16>, %rhs: tensor<2x4xf16>,
    %init: tensor<2x3xui8>) -> tensor<2x3xui8> {
  // expected-error@+1 {{lhs and rhs shapes are not broadcast compatible}}
  %result = hipsr.equal(%ctx)
      ins(%lhs, %rhs : tensor<2x3xf16>, tensor<2x4xf16>)
      outs(%init : tensor<2x3xui8>) : tensor<2x3xui8>
  return %result : tensor<2x3xui8>
}

// -----

// The output holds the broadcast of the two operand shapes.
func.func @equal_output_shape(
    %ctx: !hipsr.context, %lhs: tensor<4x1xf16>, %rhs: tensor<3xf16>,
    %init: tensor<4x1xui8>) -> tensor<4x1xui8> {
  // expected-error@+1 {{output shape must be the broadcast of lhs and rhs}}
  %result = hipsr.equal(%ctx)
      ins(%lhs, %rhs : tensor<4x1xf16>, tensor<3xf16>)
      outs(%init : tensor<4x1xui8>) : tensor<4x1xui8>
  return %result : tensor<4x1xui8>
}

// -----

// Both operands are compared on the device.
func.func @equal_host_operand(
    %ctx: !hipsr.context,
    %lhs: memref<2x3xf16, #hipsr.mem<device>>,
    %rhs: memref<2x3xf16, #hipsr.mem<host>>,
    %init: memref<2x3xui8, #hipsr.mem<device>>) {
  // expected-error@+1 {{operand #2 must be ranked tensor or device memref}}
  hipsr.equal(%ctx)
      ins(%lhs, %rhs : memref<2x3xf16, #hipsr.mem<device>>,
                       memref<2x3xf16, #hipsr.mem<host>>)
      outs(%init : memref<2x3xui8, #hipsr.mem<device>>)
  return
}
