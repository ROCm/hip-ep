// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Transpose forms the conversion rejects for good. Forms that are only
// unimplemented are not here, and a bad perm is the hipsr.transpose verifier's,
// covered by Dialect/Hipsr/IR/transpose.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --onnx-dialect=modeled --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// An unranked type carries no memory space, so an unranked input can never be
// on the device, and it has no rank to spell out a defaulted perm with.
func.func @unranked_input(%ctx: !hipsr.context,
                          %input: tensor<*xf16>) -> tensor<3x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Transpose'}}
  %0 = "onnx.Transpose"(%input) {perm = [1, 0]}
      : (tensor<*xf16>) -> tensor<3x2xf16>
  return %0 : tensor<3x2xf16>
}
