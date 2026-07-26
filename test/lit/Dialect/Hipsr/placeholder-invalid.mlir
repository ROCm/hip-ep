// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

// An unranked result is rejected by the ODS type constraint.
func.func @unranked_rejected() {
  // expected-error @+1 {{op result #0 must be variadic of ranked tensor of any type values}}
  %0 = "hipsr.placeholder"() : () -> tensor<*xf16>
  return
}

// -----

// A placeholder must produce at least one init value.
func.func @zero_result_placeholder() {
  // expected-error @+1 {{must produce at least one tensor DPS init}}
  "hipsr.placeholder"() : () -> ()
  return
}

// -----

// Every placeholder result must be used.
func.func @unused_placeholder() {
  // expected-error @+1 {{requires each result to have exactly one use}}
  %init = hipsr.placeholder : tensor<4x8xf16>
  return
}

// -----

// A placeholder result may only be used by a destination-style operation.
func.func @non_dps_placeholder_use() -> tensor<?x8xf16> {
  // expected-error @+1 {{requires each result to be used as a DPS init of a hipsr operation}}
  %init = hipsr.placeholder : tensor<4x8xf16>
  %result = tensor.cast %init : tensor<4x8xf16> to tensor<?x8xf16>
  return %result : tensor<?x8xf16>
}

// -----

// A placeholder may only initialize an operation in the hipsr dialect.
func.func @non_hipsr_dps_use(%value: f32) -> tensor<4x8xf32> {
  // expected-error @+1 {{requires each result to be used as a DPS init of a hipsr operation}}
  %init = hipsr.placeholder : tensor<4x8xf32>
  %result = linalg.fill ins(%value : f32)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// -----

// A placeholder cannot be a destination-style input.
func.func @dps_input_placeholder(%ctx: !hipsr.context) -> tensor<4x8xf16> {
  // expected-error @+1 {{requires each result to be used as a DPS init of a hipsr operation}}
  %input = hipsr.placeholder : tensor<4x8xf32>
  %init = hipsr.placeholder : tensor<4x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// A placeholder result cannot initialize multiple operations.
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

// All results of one placeholder must initialize the same operation.
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
