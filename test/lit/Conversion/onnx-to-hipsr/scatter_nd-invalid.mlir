// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.ScatterND forms the conversion rejects for good. Forms that are only
// unimplemented are not here; TODOs mark those in the conversion. Operand
// shapes and types are the hipsr.scatter_nd verifier's, in
// Dialect/Hipsr/IR/scatter_nd.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --onnx-dialect=modeled --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// Reducing a duplicate index into its destination is a read-modify-write
// rather than an overwrite, so it needs an operation of its own. The other
// three reducing modes are rejected the same way.
func.func @reduction_add(%ctx: !hipsr.context, %data: tensor<4x2xf16>,
                         %ids: tensor<5x2xi64>,
                         %updates: tensor<5xf16>) -> tensor<4x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.ScatterND'}}
  %0 = "onnx.ScatterND"(%data, %ids, %updates) {reduction = "add"}
      : (tensor<4x2xf16>, tensor<5x2xi64>, tensor<5xf16>) -> tensor<4x2xf16>
  return %0 : tensor<4x2xf16>
}
