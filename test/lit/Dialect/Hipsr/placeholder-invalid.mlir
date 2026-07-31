// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

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
  ^bb0(%input_shape: !shape.shape):
    // expected-error @+1 {{using value defined outside the region}}
    %captured_shape = shape.shape_of %input : tensor<?x8xf32> -> !shape.shape
    hipsr.shape_yield %captured_shape : !shape.shape
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
    %shape0 = "shape.from_extents"() : () -> !shape.shape
    hipsr.shape_yield %shape0 : !shape.shape
  ^bb1:
    %shape1 = "shape.from_extents"() : () -> !shape.shape
    hipsr.shape_yield %shape1 : !shape.shape
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
  ^bb0(%input_shape: !shape.shape):
    llvm.unreachable
  }) {type = #hipsr.placeholder_type<normal>}
      : (!hipsr.context, tensor<4x8xf32>) -> tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Normal regions have one shape argument per placeholder input.
func.func @shape_region_argument_count(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{shape region block argument count must match the 'normal' layout; expected 1, got 2}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%shape0: !shape.shape, %shape1: !shape.shape):
    hipsr.shape_yield %shape0 : !shape.shape
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Normal arguments carry shapes rather than the placeholder operand types.
func.func @normal_argument_type(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{shape region block argument 0 type 'tensor<4x8xf32>' does not match expected type '!shape.shape'}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%shape_input: tensor<4x8xf32>):
    %shape = shape.shape_of %shape_input
        : tensor<4x8xf32> -> !shape.shape
    hipsr.shape_yield %shape : !shape.shape
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Empty placeholders must provide every shape input used by the recipe.
func.func @missing_cast_shape_input(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{empty shape region requires at least 1 shape-graph input(s) for hipsr.cast consumer; got 0}}
  %init = hipsr.placeholder(%ctx)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Populated placeholders require exactly one input per shaped DPS input.
func.func @populated_shape_input_count(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>,
    %extra: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{populated shape region requires 1 shape-graph input(s) for hipsr.cast consumer; got 2}}
  %init = hipsr.placeholder(%ctx)
      ins(%input, %extra : tensor<4x8xf32>, tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%input_shape: !shape.shape, %extra_shape: !shape.shape):
    hipsr.shape_yield %input_shape : !shape.shape
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Barrier arguments preserve the original tensor types after context.
func.func @barrier_argument_type(
    %ctx: !hipsr.context, %input: tensor<?x3xf16>, %shape: tensor<2xi64>)
    -> tensor<?x?xf16> {
  // expected-error @+1 {{shape region block argument 1 type '!shape.shape' does not match expected type 'tensor<?x3xf16>'}}
  %init = hipsr.placeholder(%ctx)
      ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
      {type = #hipsr.placeholder_type<barrier>} : tensor<?x?xf16>
      shape_region {
  ^bb0(%shape_ctx: !hipsr.context, %input_shape: !shape.shape,
       %requested_shape: tensor<2xi64>):
    hipsr.shape_yield %input_shape : !shape.shape
  }
  %result = hipsr.expand(%ctx)
      ins(%input, %shape : tensor<?x3xf16>, tensor<2xi64>)
      outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
  return %result : tensor<?x?xf16>
}

// -----

// A populated region yields one shape per placeholder result.
func.func @shape_yield_count(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%input_shape: !shape.shape):
    // expected-error @+1 {{must yield one !shape.shape per enclosing placeholder result; expected 1, got 0}}
    "hipsr.shape_yield"() : () -> ()
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Shape yield operands must have !shape.shape type.
func.func @shape_yield_type(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
      shape_region {
  ^bb0(%input_shape: !shape.shape):
    %tensor_shape = arith.constant dense<[4, 8]> : tensor<2xi64>
    // expected-error @+1 {{operand #0 must be variadic of}}
    "hipsr.shape_yield"(%tensor_shape) : (tensor<2xi64>) -> ()
  }
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Shape yield directly terminates only placeholder regions.
func.func @shape_yield_parent() {
  "test.region_holder"() ({
    %shape = "shape.from_extents"() : () -> !shape.shape
    // expected-error @+1 {{expects parent op 'hipsr.placeholder'}}
    hipsr.shape_yield %shape : !shape.shape
  }) : () -> ()
  return
}
