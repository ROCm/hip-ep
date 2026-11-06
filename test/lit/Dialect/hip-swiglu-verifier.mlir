// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.swiglu's clamped form (activation_alpha, activation_beta,
// swiglu_limit): f32, f16 or bf16, a finite limit not below 0, and not together
// with promote_activation. The plain form is unchanged.
//
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @clamped
// CHECK:         hip.swiglu
// CHECK-SAME:    activation_alpha = 1.702000e+00 : f32
// CHECK-SAME:    activation_beta = 1.000000e+00 : f32
// CHECK-SAME:    swiglu_limit = 7.000000e+00 : f32
func.func @clamped(%ctx: !hip.context, %g: tensor<4x8xf32>, %u: tensor<4x8xf32>,
                   %init: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %r = hip.swiglu(%ctx) ins(%g, %u : tensor<4x8xf32>, tensor<4x8xf32>)
        outs(%init : tensor<4x8xf32>)
        {activation_alpha = 1.702 : f32, activation_beta = 1.0 : f32,
         swiglu_limit = 7.0 : f32} : tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @plain_f64
func.func @plain_f64(%ctx: !hip.context, %g: tensor<8xf64>, %u: tensor<8xf64>,
                     %init: tensor<8xf64>) -> tensor<8xf64> {
  %r = hip.swiglu(%ctx) ins(%g, %u : tensor<8xf64>, tensor<8xf64>)
        outs(%init : tensor<8xf64>) : tensor<8xf64>
  return %r : tensor<8xf64>
}

// -----

func.func @clamped_f64(%ctx: !hip.context, %g: tensor<8xf64>, %u: tensor<8xf64>,
                       %init: tensor<8xf64>) -> tensor<8xf64> {
  // expected-error @+1 {{the clamped form takes f32, f16 or bf16, not 'f64'}}
  %r = hip.swiglu(%ctx) ins(%g, %u : tensor<8xf64>, tensor<8xf64>)
        outs(%init : tensor<8xf64>) {swiglu_limit = 7.0 : f32} : tensor<8xf64>
  return %r : tensor<8xf64>
}

// -----

func.func @limit_negative(%ctx: !hip.context, %g: tensor<8xf32>, %u: tensor<8xf32>,
                          %init: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{swiglu_limit must be finite and not negative}}
  %r = hip.swiglu(%ctx) ins(%g, %u : tensor<8xf32>, tensor<8xf32>)
        outs(%init : tensor<8xf32>) {swiglu_limit = -1.0 : f32} : tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----

func.func @promoted(%ctx: !hip.context, %g: tensor<8xf16>, %u: tensor<8xf16>,
                    %init: tensor<8xf16>) -> tensor<8xf16> {
  // expected-error @+1 {{promote_activation does not combine with the clamped form}}
  %r = hip.swiglu(%ctx) ins(%g, %u : tensor<8xf16>, tensor<8xf16>)
        outs(%init : tensor<8xf16>)
        {promote_activation = true, activation_beta = 1.0 : f32} : tensor<8xf16>
  return %r : tensor<8xf16>
}
