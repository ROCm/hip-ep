// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.NonZero forms the conversion rejects.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --onnx-dialect=modeled --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// A scalar has no axis to name a position by.
func.func @scalar_input(%ctx: !hipsr.context,
                        %mask: tensor<i8>) -> tensor<0x?xi64> {
  // expected-error @+1 {{failed to legalize operation 'onnx.NonZero'}}
  %0 = "onnx.NonZero"(%mask) : (tensor<i8>) -> tensor<0x?xi64>
  return %0 : tensor<0x?xi64>
}

// -----

// The search compares against an integer zero, so a float input is out of
// scope.
func.func @float_input(%ctx: !hipsr.context,
                       %values: tensor<?x?xf16>) -> tensor<2x?xi64> {
  // expected-error @+1 {{failed to legalize operation 'onnx.NonZero'}}
  %0 = "onnx.NonZero"(%values) : (tensor<?x?xf16>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}

// -----

// The search runs on the device, and the only copy this emits is the count
// coming back.
func.func @host_input(%ctx: !hipsr.context,
                      %mask: tensor<?x?xi8, #hipsr.mem<host>>)
    -> tensor<2x?xi64, #hipsr.mem<host>> {
  // expected-error @+1 {{failed to legalize operation 'onnx.NonZero'}}
  %0 = "onnx.NonZero"(%mask) : (tensor<?x?xi8, #hipsr.mem<host>>)
      -> tensor<2x?xi64, #hipsr.mem<host>>
  return %0 : tensor<2x?xi64, #hipsr.mem<host>>
}

// -----

// One row per input axis leaves no room for a third.
func.func @result_rank(%ctx: !hipsr.context,
                       %mask: tensor<?x?xi8>) -> tensor<2x?x1xi64> {
  // expected-error @+1 {{failed to legalize operation 'onnx.NonZero'}}
  %0 = "onnx.NonZero"(%mask) : (tensor<?x?xi8>) -> tensor<2x?x1xi64>
  return %0 : tensor<2x?x1xi64>
}

// -----

// A position names every axis, so there is a row for each.
func.func @result_row_count(%ctx: !hipsr.context,
                            %mask: tensor<?x?xi8>) -> tensor<3x?xi64> {
  // expected-error @+1 {{failed to legalize operation 'onnx.NonZero'}}
  %0 = "onnx.NonZero"(%mask) : (tensor<?x?xi8>) -> tensor<3x?xi64>
  return %0 : tensor<3x?xi64>
}

// -----

// How many positions there are follows the values, so no result type can
// claim to know it.
func.func @static_result_columns(%ctx: !hipsr.context,
                                 %mask: tensor<?x?xi8>) -> tensor<2x4xi64> {
  // expected-error @+1 {{failed to legalize operation 'onnx.NonZero'}}
  %0 = "onnx.NonZero"(%mask) : (tensor<?x?xi8>) -> tensor<2x4xi64>
  return %0 : tensor<2x4xi64>
}
