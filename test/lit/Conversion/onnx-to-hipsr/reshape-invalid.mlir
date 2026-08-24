// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Reshape forms the conversion rejects.
//
// Every case but @non_constant_shape folds its target shape, so the rejection
// under test is the only thing standing in the way.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --onnx-dialect=modeled --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// With allowzero set, a 0 in the target shape is a literal extent instead of a
// copy of the input's. The conversion does not implement that reading, so it
// turns the attribute away rather than resolve a shape under the wrong one.
func.func @allowzero(%ctx: !hipsr.context,
                     %input: tensor<2x3xf16>) -> tensor<6xf16> {
  %shape = "onnx.Constant"() {value = dense<6> : tensor<1xi64>}
      : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 1 : si64}
      : (tensor<2x3xf16>, tensor<1xi64>) -> tensor<6xf16>
  return %0 : tensor<6xf16>
}

// -----

// With allowzero clear a 0 copies the input's extent at that axis. Resolving
// that is left out, so the entry is rejected rather than read as an extent.
func.func @zero_entry(%ctx: !hipsr.context,
                      %input: tensor<?x8x4xf16>) -> tensor<?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x8x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// Two -1 entries are one equation with two unknowns, and the element count
// recovers one. @collapse_and_expand in reshape.mlir is the same reshape with
// the second extent named.
func.func @two_inferred_extents(%ctx: !hipsr.context,
                                %input: tensor<?x8x4xf16>) -> tensor<?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<-1> : tensor<2xi64>}
      : () -> tensor<2xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x8x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// The target shape is the only thing that states the result extents, so an
// operand that stays a runtime value leaves them out of reach. Reading it would
// be a host read the conversion does not do.
func.func @non_constant_shape(%ctx: !hipsr.context, %input: tensor<2x3xf16>,
                              %shape: tensor<1xi64>) -> tensor<6xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<2x3xf16>, tensor<1xi64>) -> tensor<6xf16>
  return %0 : tensor<6xf16>
}

// -----

// A reshape preserves the element count, so 6 elements cannot become 7 by
// regrouping them.
func.func @count_mismatch(%ctx: !hipsr.context,
                          %input: tensor<2x3xf16>) -> tensor<7xf16> {
  %shape = "onnx.Constant"() {value = dense<7> : tensor<1xi64>}
      : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<2x3xf16>, tensor<1xi64>) -> tensor<7xf16>
  return %0 : tensor<7xf16>
}

// -----

// The 1-D form is built from the input's extents, so the input must be ranked.
func.func @unranked_input(%ctx: !hipsr.context,
                          %input: tensor<*xf16>) -> tensor<6xf16> {
  %shape = "onnx.Constant"() {value = dense<6> : tensor<1xi64>}
      : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<*xf16>, tensor<1xi64>) -> tensor<6xf16>
  return %0 : tensor<6xf16>
}

// -----

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%input: tensor<2x3xf16>) -> tensor<6xf16> {
  %shape = "onnx.Constant"() {value = dense<6> : tensor<1xi64>}
      : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<2x3xf16>, tensor<1xi64>) -> tensor<6xf16>
  return %0 : tensor<6xf16>
}
