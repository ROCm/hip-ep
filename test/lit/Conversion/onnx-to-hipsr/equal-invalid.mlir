// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Equal forms the conversion rejects for good. Forms that are only
// unimplemented are not here; TODOs mark those in the conversion.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --onnx-dialect=modeled --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// An unranked type carries no memory space, so an unranked operand can never
// be on the device, and it has no shape to build a device constant from.
func.func @unranked_operand(%ctx: !hipsr.context, %lhs: tensor<*xf16>,
                            %rhs: tensor<2x3xf16>) -> tensor<2x3xi1> {
  // expected-error @+1 {{failed to legalize operation 'onnx.Equal'}}
  %0 = "onnx.Equal"(%lhs, %rhs)
      : (tensor<*xf16>, tensor<2x3xf16>) -> tensor<2x3xi1>
  return %0 : tensor<2x3xi1>
}
