// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// REQUIRES: hipsr

// RUN: hip-mlir-opt %s -allow-unregistered-dialect -convert-onnx-to-hipsr -verify-diagnostics

// Converted placeholders reject dependencies on unsupported ONNX results.
func.func @unsupported_producer(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) -> tensor<4x8xf16> {
  %unsupported = "onnx.Unsupported"(%input)
      : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // expected-error @+1 {{input 0 must be a block argument or a result of hipsr.placeholder, arith.constant, or hipsr.constant; got result of 'onnx.Unsupported'}}
  %result = "onnx.Cast"(%unsupported)
      : (tensor<4x8xf32>) -> tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}
