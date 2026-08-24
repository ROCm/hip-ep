// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Reshape forms the conversion rejects.
//
// Each case folds its target shape, so the rejection under test is the only
// thing standing in the way. Forms the conversion has yet to implement are not
// here: those are marked with TODOs at the checks that turn them away, and a
// case built on one would have to go when the feature lands.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --onnx-dialect=modeled --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// Two -1 entries are one equation with two unknowns, and the element count
// recovers one. @collapse_and_expand in reshape.mlir is the same reshape with
// the second dimension named.
func.func @two_inferred_dims(%ctx: !hipsr.context,
                             %input: tensor<?x8x4xf16>) -> tensor<?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<-1> : tensor<2xi64>}
      : () -> tensor<2xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x8x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// The 1-D form is built from the input's dimensions, so it must be ranked.
func.func @unranked_input(%ctx: !hipsr.context,
                          %input: tensor<*xf16>) -> tensor<6xf16> {
  %shape = "onnx.Constant"() {value = dense<6> : tensor<1xi64>}
      : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Reshape'}}
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<*xf16>, tensor<1xi64>) -> tensor<6xf16>
  return %0 : tensor<6xf16>
}
