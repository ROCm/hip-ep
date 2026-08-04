// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// UNSUPPORTED: true

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics -allow-unregistered-dialect %s

// Results must be ranked tensors.
func.func @unranked_rejected(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) {
  // expected-error @+1 {{op result #0 must be variadic of ranked tensor of any type values}}
  %0 = "hipsr.placeholder"(%ctx, %input) ({})
      {type = #hipsr.placeholder_type<normal>}
      : (!hipsr.context, tensor<4x8xf32>) -> tensor<*xf16>
  return
}

// -----

// Inputs must be ranked tensors.
func.func @unranked_shape_operand_rejected(
    %ctx: !hipsr.context, %input: tensor<*xf32>) {
  // expected-error @+1 {{op operand #1 must be variadic of ranked tensor of any type values}}
  %0 = "hipsr.placeholder"(%ctx, %input) ({})
      {type = #hipsr.placeholder_type<normal>}
      : (!hipsr.context, tensor<*xf32>) -> tensor<4x8xf16>
  return
}

// -----

// Zero-result placeholders are invalid.
func.func @zero_result_placeholder(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) {
  // expected-error @+1 {{must produce at least one tensor DPS init}}
  "hipsr.placeholder"(%ctx, %input) ({})
      {type = #hipsr.placeholder_type<normal>}
      : (!hipsr.context, tensor<4x8xf32>) -> ()
  return
}

// -----

// Placeholder type is required even while its shape region has zero blocks.
func.func @missing_type(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{requires attribute 'type'}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>) : tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Unused placeholder results are invalid.
func.func @unused_placeholder(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) {
  // expected-error @+1 {{requires each result to initialize exactly one hipsr operation}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  return
}

// -----

// Uses other than placeholder inputs and HIPSR DPS inits are invalid.
func.func @non_dps_placeholder_use(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<?x8xf16> {
  // expected-error @+1 {{requires each result use to be a placeholder input or a DPS init of a hipsr operation}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %result = tensor.cast %init : tensor<4x8xf16> to tensor<?x8xf16>
  return %result : tensor<?x8xf16>
}

// -----

// DPS init uses must belong to HIPSR ops.
func.func @non_hipsr_dps_use(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>,
    %value: f32) -> tensor<4x8xf32> {
  // expected-error @+1 {{requires each result use to be a placeholder input or a DPS init of a hipsr operation}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32>
  %result = linalg.fill ins(%value : f32)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// -----

// A placeholder cannot be a DPS input.
func.func @dps_input_placeholder(
    %ctx: !hipsr.context, %source: tensor<4x8xf32>,
    %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error @+1 {{requires each result use to be a placeholder input or a DPS init of a hipsr operation}}
  %input = hipsr.placeholder(%ctx)
      ins(%source : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Each result can be a DPS init for only one op.
func.func @shared_placeholder(%ctx: !hipsr.context,
                              %input: tensor<4x8xf32>)
    -> (tensor<4x8xf16>, tensor<4x8xf16>) {
  // expected-error @+1 {{requires each result to initialize exactly one hipsr operation}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  %second = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %first, %second : tensor<4x8xf16>, tensor<4x8xf16>
}

// -----

// All results must be DPS init values for the same op.
func.func @split_placeholder_consumers(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>)
    -> (tensor<4x8xf16>, tensor<4x8xf16>) {
  // expected-error @+1 {{requires all results to initialize the same hipsr operation}}
  %inits:2 = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>}
      : tensor<4x8xf16>, tensor<4x8xf16>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%inits#0 : tensor<4x8xf16>) : tensor<4x8xf16>
  %second = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%inits#1 : tensor<4x8xf16>) : tensor<4x8xf16>
  return %first, %second : tensor<4x8xf16>, tensor<4x8xf16>
}

// -----

// A placeholder result must match the corresponding consumer result type.
func.func @result_type_mismatch(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{result 0 type 'tensor<4x8xf16>' must match consumer result type 'tensor<4x8xf32>'}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  // expected-error @+1 {{expected type of operand #2 ('tensor<4x8xf16>') to match type of corresponding result ('tensor<4x8xf32>')}}
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// -----

// Placeholder inputs always follow the HIPSR context.
func.func @missing_context(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{expected 1 or more operands, but found 0}}
  %init = "hipsr.placeholder"() ({})
      {type = #hipsr.placeholder_type<normal>} : () -> tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// A context after a data operand does not satisfy the context-first contract.
func.func @nonleading_context(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{operand #0 must be Opaque hipsr execution context}}
  %init = "hipsr.placeholder"(%input, %ctx) ({})
      {type = #hipsr.placeholder_type<normal>}
      : (tensor<4x8xf32>, !hipsr.context) -> tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Shape dependencies use the producer's placeholder, not its data result.
func.func @data_result_shape_input(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %first_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%first_init : tensor<4x8xf16>) : tensor<4x8xf16>
  // expected-error @+1 {{input 0 must be a block argument or a result of hipsr.placeholder, arith.constant, or hipsr.constant; got result of 'hipsr.cast'}}
  %second_init = hipsr.placeholder(%ctx)
      ins(%first : tensor<4x8xf16>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf32>
  %second = hipsr.cast(%ctx) ins(%first : tensor<4x8xf16>)
      outs(%second_init : tensor<4x8xf32>) : tensor<4x8xf32>
  return %second : tensor<4x8xf32>
}

// -----

// A value from an unconverted operation cannot root the shape graph.
func.func @unconverted_shape_input(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  %unconverted = "onnx.Unsupported"(%input)
      : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // expected-error @+1 {{input 0 must be a block argument or a result of hipsr.placeholder, arith.constant, or hipsr.constant; got result of 'onnx.Unsupported'}}
  %init = hipsr.placeholder(%ctx)
      ins(%unconverted : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%unconverted : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Placeholder shape regions cannot capture values from their parent scope.
func.func @shape_region_capture(
    %ctx: !hipsr.context, %input: tensor<?x8xf32>) -> tensor<?x8xf16> {
  // expected-note @+1 {{required by region isolation constraints}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
      shape_region {
  ^bb0(%shape_ctx: !hipsr.context, %shape_input: tensor<?x8xf32>):
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{using value defined outside the region}}
    %d0 = tensor.dim %input, %c0 : tensor<?x8xf32>
    %c8 = arith.constant 8 : index
    hipsr.shape_yield (%d0, %c8) : [f16]
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %result : tensor<?x8xf16>
}

// -----

// A placeholder shape region may have at most one block.
func.func @two_shape_region_blocks(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
    hipsr.shape_yield () : [f16]
  ^bb1:
    hipsr.shape_yield () : [f16]
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// A placeholder shape region must end with hipsr.shape_yield.
func.func @wrong_shape_region_terminator(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+2 {{expects regions to end with 'hipsr.shape_yield'}}
  // expected-note @+1 {{in custom textual format, the absence of terminator implies 'hipsr.shape_yield'}}
  %init = "hipsr.placeholder"(%ctx, %input) ({
  ^bb0(%shape_ctx: !hipsr.context, %shape_input: tensor<4x8xf32>):
    llvm.unreachable
  }) {type = #hipsr.placeholder_type<normal>}
      : (!hipsr.context, tensor<4x8xf32>) -> tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// A populated shape region has one block argument per placeholder operand.
func.func @shape_region_argument_count(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{shape region block argument count must match operand count; expected 2, got 1}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%shape_ctx: !hipsr.context):
    hipsr.shape_yield () : [f16]
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}
