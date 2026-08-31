// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Verifier and region-trait rules for hipsr.placeholder. A case without CHECK
// lines is invalid IR that trips one rule.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// A placeholder exists to hold outs values, so it must produce one. Only the
// generic form can write an empty result list.
func.func @no_result(%ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>) {
  // expected-error @+1 {{must produce at least one tensor outs operand}}
  "hipsr.placeholder"(%ctx, %input) ({})
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : (!hipsr.context, tensor<4x8xf32, #hipsr.mem<device>>) -> ()
  return
}

// -----
// A shape input comes from the producer's placeholder, never from its data
// result.
func.func @data_result_input(
    %ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf32, #hipsr.mem<device>> {
  %first_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16, #hipsr.mem<device>>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%first_init : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
  // expected-error @+1 {{input 0 must be a block argument or a result of hipsr.placeholder, arith.constant, or hipsr.constant; got result of 'hipsr.cast'}}
  %second_init = hipsr.placeholder(%ctx)
      ins(%first : tensor<4x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32, #hipsr.mem<device>>
  %second = hipsr.cast(%ctx) ins(%first : tensor<4x8xf16, #hipsr.mem<device>>)
      outs(%second_init : tensor<4x8xf32, #hipsr.mem<device>>) : tensor<4x8xf32, #hipsr.mem<device>>
  return %second : tensor<4x8xf32, #hipsr.mem<device>>
}

// -----
// A placeholder that skips an operand its consumer reads can land in an
// earlier pool domain than the consumer.
func.func @input_missing_from_placeholder(
    %ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf32, #hipsr.mem<device>> {
  %first_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32, #hipsr.mem<device>>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%first_init : tensor<4x8xf32, #hipsr.mem<device>>) : tensor<4x8xf32, #hipsr.mem<device>>
  // expected-error @+1 {{must read the shape-graph value of input 1 of its consumer 'hipsr.add'}}
  %sum_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32, #hipsr.mem<device>>
  %sum = hipsr.add(%ctx)
      ins(%input, %first : tensor<4x8xf32, #hipsr.mem<device>>, tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%sum_init : tensor<4x8xf32, #hipsr.mem<device>>) : tensor<4x8xf32, #hipsr.mem<device>>
  return %sum : tensor<4x8xf32, #hipsr.mem<device>>
}

// -----
// A result that fills no outs slot initializes nothing.
func.func @unused_result(%ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>) {
  // expected-error @+1 {{requires each result to initialize exactly one hipsr operation}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16, #hipsr.mem<device>>
  return
}

// -----
// The other half of the same rule: one result cannot initialize two ops,
// because both would write the same buffer.
func.func @shared_result(%ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>)
    -> (tensor<4x8xf16, #hipsr.mem<device>>, tensor<4x8xf16, #hipsr.mem<device>>) {
  // expected-error @+1 {{requires each result to initialize exactly one hipsr operation}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16, #hipsr.mem<device>>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
  %second = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
  return %first, %second : tensor<4x8xf16, #hipsr.mem<device>>, tensor<4x8xf16, #hipsr.mem<device>>
}

// -----
// A linalg.fill init is a destination, but the pool-domain passes only place
// hipsr ops, so a non-hipsr consumer is rejected.
func.func @non_hipsr_outs(%ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>,
                          %value: f32) -> tensor<4x8xf32, #hipsr.mem<device>> {
  // expected-error @+1 {{requires each result use to be a placeholder input, pool-domain yield, or an outs operand of a hipsr operation}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32, #hipsr.mem<device>>
  %result = linalg.fill ins(%value : f32)
      outs(%init : tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf32, #hipsr.mem<device>>
  return %result : tensor<4x8xf32, #hipsr.mem<device>>
}

// -----
// A normal shape region takes one extent tensor per input.
func.func @shape_region_argument_count(
    %ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf16, #hipsr.mem<device>> {
  // expected-error @+1 {{shape region block argument count does not match the placeholder type layout; expected 1, got 0}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16, #hipsr.mem<device>>
      shape_region {
    %shape = shape.const_shape [4, 8] : tensor<2xindex>
    hipsr.shape_yield %shape : tensor<2xindex>
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
  return %result : tensor<4x8xf16, #hipsr.mem<device>>
}

// -----
// A normal region gets shapes, not the input tensors. Only a barrier region
// takes the tensor types, led by ctx.
func.func @shape_region_argument_type(
    %ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf16, #hipsr.mem<device>> {
  // expected-error @+1 {{shape region block argument 0 type 'tensor<4x8xf32, #hipsr.mem<device>>' does not match expected type 'tensor<2xindex>'}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16, #hipsr.mem<device>>
      shape_region {
  ^bb0(%input_shape: tensor<4x8xf32, #hipsr.mem<device>>):
    %shape = shape.const_shape [4, 8] : tensor<2xindex>
    hipsr.shape_yield %shape : tensor<2xindex>
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
  return %result : tensor<4x8xf16, #hipsr.mem<device>>
}

// -----
// The mirror case: a barrier region reads the inputs, so ctx leads and the
// shapes of a normal region do not fit.
func.func @barrier_shape_region_layout(
    %ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf16, #hipsr.mem<device>> {
  // expected-error @+1 {{shape region block argument 0 type 'tensor<2xindex>' does not match expected type '!hipsr.context'}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4x8xf16, #hipsr.mem<device>>
      shape_region {
  ^bb0(%input_shape: tensor<2xindex>, %data: tensor<4x8xf32, #hipsr.mem<device>>):
    %shape = shape.const_shape [4, 8] : tensor<2xindex>
    hipsr.shape_yield %shape : tensor<2xindex>
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
  return %result : tensor<4x8xf16, #hipsr.mem<device>>
}

// -----
// The region is IsolatedFromAbove: shapes must come from its block arguments,
// so the region stays valid wherever a pass moves the placeholder.
func.func @shape_region_capture(
    %ctx: !hipsr.context, %input: tensor<?x8xf32, #hipsr.mem<device>>) -> tensor<?x8xf16, #hipsr.mem<device>> {
  // expected-note @+1 {{required by region isolation constraints}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
      shape_region {
  ^bb0(%input_shape: tensor<2xindex>):
    // expected-error @+1 {{using value defined outside the region}}
    %captured = shape.shape_of %input : tensor<?x8xf32, #hipsr.mem<device>> -> tensor<2xindex>
    hipsr.shape_yield %captured : tensor<2xindex>
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>
  return %result : tensor<?x8xf16, #hipsr.mem<device>>
}

// -----
// The shape graph has no control flow, so the region holds at most one block.
func.func @two_shape_region_blocks(
    %ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf16, #hipsr.mem<device>> {
  // expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16, #hipsr.mem<device>>
      shape_region {
  ^bb0(%input_shape: tensor<2xindex>):
    hipsr.shape_yield %input_shape : tensor<2xindex>
  ^bb1:
    %shape = shape.const_shape [4, 8] : tensor<2xindex>
    hipsr.shape_yield %shape : tensor<2xindex>
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
  return %result : tensor<4x8xf16, #hipsr.mem<device>>
}

// -----
// Shapes leave the region only through hipsr.shape_yield.
func.func @wrong_shape_region_terminator(
    %ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf16, #hipsr.mem<device>> {
  // expected-error @+2 {{expects regions to end with 'hipsr.shape_yield', found 'llvm.unreachable'}}
  // expected-note @+1 {{in custom textual format, the absence of terminator implies 'hipsr.shape_yield'}}
  %init = "hipsr.placeholder"(%ctx, %input) ({
  ^bb0(%input_shape: tensor<2xindex>):
    llvm.unreachable
  }) {placeholder_type = #hipsr.placeholder_type<normal>}
      : (!hipsr.context, tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf16, #hipsr.mem<device>>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
  return %result : tensor<4x8xf16, #hipsr.mem<device>>
}

// -----
// Arith and hipsr constants are shape-graph roots, so a placeholder can take
// its shape straight from a constant.
// CHECK-LABEL: func.func @constant_shape_roots(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[ARITH_SHAPE:.+]] = arith.constant dense<[4, 8]> : tensor<2xi64>
// CHECK-NEXT: %[[HIPSR_SHAPE:.+]] = hipsr.constant {value = dense<[4, 8]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[ARITH_SHAPE]], %[[HIPSR_SHAPE]] : tensor<2xi64>, tensor<2xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<4x8xf32, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
// CHECK-NEXT: return %[[RESULT]] : tensor<4x8xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @constant_shape_roots(
    %ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf16, #hipsr.mem<device>> {
  %arith_shape = arith.constant dense<[4, 8]> : tensor<2xi64>
  %hipsr_shape = hipsr.constant
      {value = dense<[4, 8]> : tensor<2xi64>} : tensor<2xi64, #hipsr.mem<device>>
  %init = hipsr.placeholder(%ctx)
      ins(%arith_shape, %hipsr_shape : tensor<2xi64>, tensor<2xi64, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16, #hipsr.mem<device>>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
  return %result : tensor<4x8xf16, #hipsr.mem<device>>
}

// -----
// A placeholder that reads a value its consumer skips can land in a later pool
// domain than the consumer.
func.func @input_missing_from_consumer(
    %ctx: !hipsr.context, %input: tensor<4x8xf32, #hipsr.mem<device>>) -> tensor<4x8xf32, #hipsr.mem<device>> {
  %extra_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16, #hipsr.mem<device>>
  %extra = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%extra_init : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
  %first_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16, #hipsr.mem<device>>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%first_init : tensor<4x8xf16, #hipsr.mem<device>>) : tensor<4x8xf16, #hipsr.mem<device>>
  // expected-error @+1 {{input 1 has no matching operand in its consumer 'hipsr.cast'}}
  %result_init = hipsr.placeholder(%ctx)
      ins(%first_init, %extra_init
          : tensor<4x8xf16, #hipsr.mem<device>>, tensor<4x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32, #hipsr.mem<device>>
  %result = hipsr.cast(%ctx) ins(%first : tensor<4x8xf16, #hipsr.mem<device>>)
      outs(%result_init : tensor<4x8xf32, #hipsr.mem<device>>) : tensor<4x8xf32, #hipsr.mem<device>>
  return %result : tensor<4x8xf32, #hipsr.mem<device>>
}
