// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Verifier and region-trait rules for hipsr.placeholder. Every case but the
// last is invalid IR that trips one rule.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// A placeholder exists to hold outs values, so it must produce one. Only the
// generic form can write an empty result list.
func.func @no_result(%ctx: !hipsr.context, %input: tensor<4x8xf32>) {
  // expected-error @+1 {{must produce at least one tensor outs operand}}
  "hipsr.placeholder"(%ctx, %input) ({})
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : (!hipsr.context, tensor<4x8xf32>) -> ()
  return
}

// -----
// A shape input comes from the producer's placeholder, never from its data
// result.
func.func @data_result_input(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %first_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%first_init : tensor<4x8xf16>) : tensor<4x8xf16>
  // expected-error @+1 {{input 0 must be a block argument or a result of hipsr.placeholder, arith.constant, or hipsr.constant; got result of 'hipsr.cast'}}
  %second_init = hipsr.placeholder(%ctx)
      ins(%first : tensor<4x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32>
  %second = hipsr.cast(%ctx) ins(%first : tensor<4x8xf16>)
      outs(%second_init : tensor<4x8xf32>) : tensor<4x8xf32>
  return %second : tensor<4x8xf32>
}

// -----
// A result that fills no outs slot initializes nothing.
func.func @unused_result(%ctx: !hipsr.context, %input: tensor<4x8xf32>) {
  // expected-error @+1 {{requires each result to initialize exactly one hipsr operation}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  return
}

// -----
// The other half of the same rule: one result cannot initialize two ops,
// because both would write the same buffer.
func.func @shared_result(%ctx: !hipsr.context, %input: tensor<4x8xf32>)
    -> (tensor<4x8xf16>, tensor<4x8xf16>) {
  // expected-error @+1 {{requires each result to initialize exactly one hipsr operation}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  %second = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %first, %second : tensor<4x8xf16>, tensor<4x8xf16>
}

// -----
// A linalg.fill init is a destination, but the pool-domain passes only place
// hipsr ops, so a non-hipsr consumer is rejected.
func.func @non_hipsr_outs(%ctx: !hipsr.context, %input: tensor<4x8xf32>,
                          %value: f32) -> tensor<4x8xf32> {
  // expected-error @+1 {{requires each result use to be a placeholder input, pool-domain yield, or an outs operand of a hipsr operation}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32>
  %result = linalg.fill ins(%value : f32)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// -----
// A DPS consumer ties each outs operand to a result, so the placeholder type
// must match that result. Non-DPS consumers such as hipsr.compute are exempt.
func.func @result_type_mismatch(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{result 0 type 'tensor<4x8xf16>' must match consumer result type 'tensor<4x8xf32>'}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  // expected-error @+1 {{expected type of operand #2 ('tensor<4x8xf16>') to match type of corresponding result ('tensor<4x8xf32>')}}
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// -----
// A normal shape region takes one !shape.shape per input.
func.func @shape_region_argument_count(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{shape region block argument count does not match the placeholder type layout; expected 1, got 0}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
    %shape = shape.const_shape [4, 8] : !shape.shape
    hipsr.shape_yield %shape : !shape.shape
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----
// A normal region gets shapes, not the input tensors. Only a barrier region
// takes the tensor types, led by ctx.
func.func @shape_region_argument_type(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{shape region block argument 0 type 'tensor<4x8xf32>' does not match expected type '!shape.shape'}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%input_shape: tensor<4x8xf32>):
    %shape = shape.const_shape [4, 8] : !shape.shape
    hipsr.shape_yield %shape : !shape.shape
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----
// The mirror case: a barrier region reads the inputs, so ctx leads and the
// shapes of a normal region do not fit.
func.func @barrier_shape_region_layout(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{shape region block argument 0 type '!shape.shape' does not match expected type '!hipsr.context'}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%input_shape: !shape.shape, %data: tensor<4x8xf32>):
    %shape = shape.const_shape [4, 8] : !shape.shape
    hipsr.shape_yield %shape : !shape.shape
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----
// The region is IsolatedFromAbove: shapes must come from its block arguments,
// so the region stays valid wherever a pass moves the placeholder.
func.func @shape_region_capture(
    %ctx: !hipsr.context, %input: tensor<?x8xf32>) -> tensor<?x8xf16> {
  // expected-note @+1 {{required by region isolation constraints}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
      shape_region {
  ^bb0(%input_shape: !shape.shape):
    // expected-error @+1 {{using value defined outside the region}}
    %captured = shape.shape_of %input : tensor<?x8xf32> -> !shape.shape
    hipsr.shape_yield %captured : !shape.shape
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %result : tensor<?x8xf16>
}

// -----
// The shape graph has no control flow, so the region holds at most one block.
func.func @two_shape_region_blocks(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%input_shape: !shape.shape):
    hipsr.shape_yield %input_shape : !shape.shape
  ^bb1:
    %shape = shape.const_shape [4, 8] : !shape.shape
    hipsr.shape_yield %shape : !shape.shape
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----
// Shapes leave the region only through hipsr.shape_yield.
func.func @wrong_shape_region_terminator(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+2 {{expects regions to end with 'hipsr.shape_yield', found 'llvm.unreachable'}}
  // expected-note @+1 {{in custom textual format, the absence of terminator implies 'hipsr.shape_yield'}}
  %init = "hipsr.placeholder"(%ctx, %input) ({
  ^bb0(%input_shape: !shape.shape):
    llvm.unreachable
  }) {placeholder_type = #hipsr.placeholder_type<normal>}
      : (!hipsr.context, tensor<4x8xf32>) -> tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----
// Arith and hipsr constants are shape-graph roots, so a placeholder can take
// its shape straight from a constant.
// CHECK-LABEL: func.func @constant_shape_roots(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<4x8xf32>) -> tensor<4x8xf16> {
// CHECK-NEXT: %[[ARITH_SHAPE:.+]] = arith.constant dense<[4, 8]> : tensor<2xi64>
// CHECK-NEXT: %[[HIPSR_SHAPE:.+]] = hipsr.constant {value = dense<[4, 8]> : tensor<2xi64>} : tensor<2xi64>
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[ARITH_SHAPE]], %[[HIPSR_SHAPE]] : tensor<2xi64>, tensor<2xi64>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<4x8xf32>) outs(%[[INIT]] : tensor<4x8xf16>) : tensor<4x8xf16>
// CHECK-NEXT: return %[[RESULT]] : tensor<4x8xf16>
// CHECK-NEXT: }
func.func @constant_shape_roots(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  %arith_shape = arith.constant dense<[4, 8]> : tensor<2xi64>
  %hipsr_shape = hipsr.constant
      {value = dense<[4, 8]> : tensor<2xi64>} : tensor<2xi64>
  %init = hipsr.placeholder(%ctx)
      ins(%arith_shape, %hipsr_shape : tensor<2xi64>, tensor<2xi64>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}
