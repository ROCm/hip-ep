// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// Every top-level ONNX operation requires a conversion pattern.
func.func @unconverted_onnx(
    %input: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Unsupported'}}
  %result = "onnx.Unsupported"(%input)
      : (tensor<2x3xf16>) -> tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

// Helper operations from other dialects must be nested in hipsr.compute.
func.func @foreign_top_level() -> tensor<2xf32> {
  // expected-error @+1 {{failed to legalize operation 'tensor.empty'}}
  %result = tensor.empty() : tensor<2xf32>
  return %result : tensor<2xf32>
}

// -----

// Only arith.constant is legal from the arith dialect at top level.
func.func @foreign_arith_top_level(%lhs: f32, %rhs: f32) -> f32 {
  // expected-error @+1 {{failed to legalize operation 'arith.addf'}}
  %result = arith.addf %lhs, %rhs : f32
  return %result : f32
}

// -----

// ONNX operations remain illegal at any nesting depth inside hipsr.compute.
func.func @nested_onnx(
    %ctx: !hipsr.context, %input: tensor<2x3xf16, #hipsr.mem<device>>,
    %init: tensor<6xf16, #hipsr.mem<device>>) -> tensor<6xf16, #hipsr.mem<device>> {
  %result = hipsr.compute(%ctx) ins(%input : tensor<2x3xf16, #hipsr.mem<device>>)
                                  outs(%init : tensor<6xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %body_input: tensor<2x3xf16, #hipsr.mem<device>>,
       %body_init: tensor<6xf16, #hipsr.mem<device>>):
    %nested = scf.execute_region -> tensor<6xf16, #hipsr.mem<device>> {
      // expected-error @+1 {{failed to legalize operation 'onnx.Unsupported'}}
      %onnx = "onnx.Unsupported"(%body_input)
          : (tensor<2x3xf16, #hipsr.mem<device>>) -> tensor<6xf16, #hipsr.mem<device>>
      scf.yield %onnx : tensor<6xf16, #hipsr.mem<device>>
    }
    hipsr.compute_yield %nested : tensor<6xf16, #hipsr.mem<device>>
  } : tensor<6xf16, #hipsr.mem<device>>
  return %result : tensor<6xf16, #hipsr.mem<device>>
}
