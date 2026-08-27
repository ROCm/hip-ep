// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --onnx-dialect=modeled --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// A window operand is read on the host, and a bound handed in as a graph input
// sits on the device, where reading it would take a copy this conversion does
// not emit. Only a constant becomes an attribute, whichever side it lives on.
func.func @device_runtime_bound(%ctx: !hipsr.context, %data: tensor<8xf16>,
                                %ends: tensor<1xi64>) -> tensor<3xf16> {
  %starts = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{failed to legalize operation 'onnx.Slice'}}
  %0 = "onnx.Slice"(%data, %starts, %ends, %none, %none)
      : (tensor<8xf16>, tensor<1xi64>, tensor<1xi64>, none, none)
      -> tensor<3xf16>
  return %0 : tensor<3xf16>
}
