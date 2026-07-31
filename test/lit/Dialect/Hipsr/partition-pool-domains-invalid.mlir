// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics -hipsr-partition-pool-domains %s

// Multi-block functions are not supported.
// expected-error @+1 {{hipsr-partition-pool-domains only supports single-block functions}}
func.func @multi_block() {
  return
^bb1:
  return
}

// -----

// Existing top-level domains are not supported.
func.func @existing_domain(%ctx: !hipsr.context, %arg: i32) -> i32 {
  // expected-error @+1 {{hipsr-partition-pool-domains does not support existing pool domains}}
  %0 = hipsr.pool_domain(%ctx, %arg : !hipsr.context, i32) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_arg: i32):
    hipsr.pool_domain_yield %domain_arg : i32
  } -> i32
  return %0 : i32
}

// -----

// Existing nested domains are not supported.
func.func @nested_existing_domain(
    %ctx: !hipsr.context, %arg: i32) -> i32 {
  %0 = scf.execute_region -> i32 {
    // expected-error @+1 {{hipsr-partition-pool-domains does not support existing pool domains}}
    %1 = hipsr.pool_domain(%ctx, %arg : !hipsr.context, i32) {
    ^bb0(%domain_ctx: !hipsr.context, %domain_arg: i32):
      hipsr.pool_domain_yield %domain_arg : i32
    } -> i32
    scf.yield %1 : i32
  }
  return %0 : i32
}

// -----

// Each tensor DPS init needs a top-level placeholder.
func.func @non_placeholder_init(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>,
    %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error @+1 {{requires each tensor DPS init to be produced by a top-level hipsr.placeholder}}
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

// Nested placeholders are not supported.
func.func @nested_placeholder(%ctx: !hipsr.context,
                              %input: tensor<4x8xf32>)
    -> tensor<4x8xf16> {
  %result = scf.execute_region -> tensor<4x8xf16> {
    // expected-error @+1 {{must be top-level when partitioning pool domains}}
    %init = hipsr.placeholder(%ctx)
        ins(%input : tensor<4x8xf32>)
        {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
    %cast = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
        outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
    scf.yield %cast : tensor<4x8xf16>
  }
  return %result : tensor<4x8xf16>
}

// -----

// The op that uses a placeholder must be top-level.
func.func @nested_placeholder_consumer(%ctx: !hipsr.context,
                                       %input: tensor<4x8xf32>)
    -> tensor<4x8xf16> {
  // expected-error @+1 {{requires its DPS consumer to be top-level when partitioning pool domains}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16>
  %result = scf.execute_region -> tensor<4x8xf16> {
    %cast = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
        outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
    scf.yield %cast : tensor<4x8xf16>
  }
  return %result : tensor<4x8xf16>
}
