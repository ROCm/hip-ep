// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

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

// Shape operands must be ranked tensors.
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
  %init = hipsr.placeholder(
      %ctx, %input : !hipsr.context, tensor<4x8xf32>) : tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Unused placeholder results are invalid.
func.func @unused_placeholder(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) {
  // expected-error @+1 {{requires each result to have exactly one use}}
  %init = hipsr.placeholder(
      %ctx, %input : !hipsr.context, tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  return
}

// -----

// Placeholder results must be DPS init values.
func.func @non_dps_placeholder_use(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<?x8xf16> {
  // expected-error @+1 {{requires each result to be used as a DPS init of a hipsr operation}}
  %init = hipsr.placeholder(
      %ctx, %input : !hipsr.context, tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %result = tensor.cast %init : tensor<4x8xf16> to tensor<?x8xf16>
  return %result : tensor<?x8xf16>
}

// -----

// Placeholder results must be used by hipsr ops.
func.func @non_hipsr_dps_use(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>,
    %value: f32) -> tensor<4x8xf32> {
  // expected-error @+1 {{requires each result to be used as a DPS init of a hipsr operation}}
  %init = hipsr.placeholder(
      %ctx, %input : !hipsr.context, tensor<4x8xf32>)
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
  // expected-error @+1 {{requires each result to be used as a DPS init of a hipsr operation}}
  %input = hipsr.placeholder(
      %ctx, %source : !hipsr.context, tensor<4x8xf32>)
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
  // expected-error @+1 {{requires each result to have exactly one use}}
  %init = hipsr.placeholder(
      %ctx, %input : !hipsr.context, tensor<4x8xf32>)
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
  %inits:2 = hipsr.placeholder(
      %ctx, %input : !hipsr.context, tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>}
      : tensor<4x8xf16>, tensor<4x8xf16>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%inits#0 : tensor<4x8xf16>) : tensor<4x8xf16>
  %second = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%inits#1 : tensor<4x8xf16>) : tensor<4x8xf16>
  return %first, %second : tensor<4x8xf16>, tensor<4x8xf16>
}

// -----

// Placeholder operands always start with the HIPSR context.
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
  // expected-error @+2 {{expected hipsr.context}}
  %init = hipsr.placeholder(
      %input, %ctx : tensor<4x8xf32>, !hipsr.context)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Placeholder operands mirror the consumer's complete DPS input list.
func.func @missing_data_input(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{operand count must match consumer DPS input count; expected 2, got 1}}
  %init = hipsr.placeholder(%ctx : !hipsr.context)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Matching types are insufficient: each placeholder operand mirrors the same
// SSA value used by the consumer.
func.func @wrong_data_input(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>,
    %other: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{operand 1 must match consumer DPS input 1}}
  %init = hipsr.placeholder(
      %ctx, %other : !hipsr.context, tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// A placeholder shape region may have at most one block.
func.func @two_shape_region_blocks(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
  %init = hipsr.placeholder(
      %ctx, %input : !hipsr.context, tensor<4x8xf32>)
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

// A populated shape region has one block argument per placeholder operand.
func.func @shape_region_argument_count(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{shape region block argument count must match operand count; expected 2, got 1}}
  %init = hipsr.placeholder(
      %ctx, %input : !hipsr.context, tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%shape_ctx: !hipsr.context):
    hipsr.shape_yield () : [f16]
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Placeholder shape regions terminate with hipsr.shape_yield.
func.func @wrong_shape_region_terminator(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{shape region must terminate with hipsr.shape_yield}}
  %init = hipsr.placeholder(
      %ctx, %input : !hipsr.context, tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%shape_ctx: !hipsr.context, %shape_input: tensor<4x8xf32>):
    cf.br ^bb0(%shape_ctx, %shape_input
        : !hipsr.context, tensor<4x8xf32>)
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}
