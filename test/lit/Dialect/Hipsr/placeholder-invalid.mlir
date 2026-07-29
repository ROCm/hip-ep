// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

// Results must be ranked tensors.
func.func @unranked_rejected() {
  // expected-error @+1 {{op result #0 must be variadic of ranked tensor of any type values}}
  %0 = "hipsr.placeholder"() : () -> tensor<*xf16>
  return
}

// -----

// Zero-result placeholders are invalid.
func.func @zero_result_placeholder() {
  // expected-error @+1 {{must produce at least one tensor DPS init}}
  "hipsr.placeholder"() : () -> ()
  return
}

// -----

// Unused placeholder results are invalid.
func.func @unused_placeholder() {
  // expected-error @+1 {{requires each result to have exactly one use}}
  %init = hipsr.placeholder : tensor<4x8xf16>
  return
}

// -----

// Placeholder results must be DPS init values.
func.func @non_dps_placeholder_use() -> tensor<?x8xf16> {
  // expected-error @+1 {{requires each result to be used as a DPS init of a hipsr operation}}
  %init = hipsr.placeholder : tensor<4x8xf16>
  %result = tensor.cast %init : tensor<4x8xf16> to tensor<?x8xf16>
  return %result : tensor<?x8xf16>
}

// -----

// Placeholder results must be used by hipsr ops.
func.func @non_hipsr_dps_use(%value: f32) -> tensor<4x8xf32> {
  // expected-error @+1 {{requires each result to be used as a DPS init of a hipsr operation}}
  %init = hipsr.placeholder : tensor<4x8xf32>
  %result = linalg.fill ins(%value : f32)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// -----

// A placeholder cannot be a DPS input.
func.func @dps_input_placeholder(%ctx: !hipsr.context) -> tensor<4x8xf16> {
  // expected-error @+1 {{requires each result to be used as a DPS init of a hipsr operation}}
  %input = hipsr.placeholder : tensor<4x8xf32>
  %init = hipsr.placeholder : tensor<4x8xf16>
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
  %init = hipsr.placeholder : tensor<4x8xf16>
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
  %inits:2 = hipsr.placeholder : tensor<4x8xf16>, tensor<4x8xf16>
  %first = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%inits#0 : tensor<4x8xf16>) : tensor<4x8xf16>
  %second = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%inits#1 : tensor<4x8xf16>) : tensor<4x8xf16>
  return %first, %second : tensor<4x8xf16>, tensor<4x8xf16>
}
