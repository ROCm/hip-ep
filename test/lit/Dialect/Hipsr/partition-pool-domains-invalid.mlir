// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics -hipsr-partition-pool-domains %s

// The pass only supports a single-block function body.
// expected-error @+1 {{hipsr-partition-pool-domains only supports single-block functions}}
func.func @multi_block() {
  return
^bb1:
  return
}

// -----

// Repartitioning an existing pool domain is unsupported.
func.func @existing_domain(%arg: i32) -> i32 {
  // expected-error @+1 {{hipsr-partition-pool-domains does not support existing pool domains}}
  %0 = hipsr.pool_domain(%arg : i32) {
  ^bb0(%domain_arg: i32):
    hipsr.pool_domain_yield %domain_arg : i32
  } -> i32
  return %0 : i32
}

// -----

// Every tensor DPS init must come from a dedicated placeholder.
func.func @non_placeholder_init(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>,
    %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error @+1 {{requires each tensor DPS init to be produced by a top-level hipsr.placeholder}}
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

// -----

// A placeholder result may only be used as a hipsr DPS init.
func.func @non_dps_placeholder_use() -> tensor<?x8xf16> {
  // expected-error @+1 {{requires every result to be a DPS init of one top-level hipsr operation}}
  %init = hipsr.placeholder : tensor<4x8xf16>
  %result = tensor.cast %init : tensor<4x8xf16> to tensor<?x8xf16>
  return %result : tensor<?x8xf16>
}

// -----

// Placeholders must be top-level so planning can move them with consumers.
func.func @nested_placeholder(%ctx: !hipsr.context,
                              %input: tensor<4x8xf32>)
    -> tensor<4x8xf16> {
  %result = scf.execute_region -> tensor<4x8xf16> {
    // expected-error @+1 {{must be top-level when partitioning pool domains}}
    %init = hipsr.placeholder : tensor<4x8xf16>
    %cast = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
        outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
    scf.yield %cast : tensor<4x8xf16>
  }
  return %result : tensor<4x8xf16>
}

// -----

// A placeholder must produce at least one init value.
func.func @zero_result_placeholder() {
  // expected-error @+1 {{must produce at least one tensor DPS init}}
  "hipsr.placeholder"() : () -> ()
  return
}
