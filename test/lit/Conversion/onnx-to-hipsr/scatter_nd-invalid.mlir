// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.ScatterND forms the conversion rejects.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// Reducing a duplicate index into its destination is a read-modify-write
// rather than an overwrite.
func.func @reduction_add(%ctx: !hipsr.context, %data: tensor<4x2xf16>,
                         %ids: tensor<5x2xi64>,
                         %updates: tensor<5xf16>) -> tensor<4x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.ScatterND'}}
  %0 = "onnx.ScatterND"(%data, %ids, %updates) {reduction = "add"}
      : (tensor<4x2xf16>, tensor<5x2xi64>, tensor<5xf16>) -> tensor<4x2xf16>
  return %0 : tensor<4x2xf16>
}

// -----

// The other three reducing modes are rejected the same way.
func.func @reduction_max(%ctx: !hipsr.context, %data: tensor<4x2xf16>,
                         %ids: tensor<5x2xi64>,
                         %updates: tensor<5xf16>) -> tensor<4x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.ScatterND'}}
  %0 = "onnx.ScatterND"(%data, %ids, %updates) {reduction = "max"}
      : (tensor<4x2xf16>, tensor<5x2xi64>, tensor<5xf16>) -> tensor<4x2xf16>
  return %0 : tensor<4x2xf16>
}

// -----

// Indices name positions, so they cannot be floating point.
func.func @float_indices(%ctx: !hipsr.context, %data: tensor<4x2xf16>,
                         %ids: tensor<5x2xf32>,
                         %updates: tensor<5xf16>) -> tensor<4x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.ScatterND'}}
  %0 = "onnx.ScatterND"(%data, %ids, %updates) {reduction = "none"}
      : (tensor<4x2xf16>, tensor<5x2xf32>, tensor<5xf16>) -> tensor<4x2xf16>
  return %0 : tensor<4x2xf16>
}

// -----

// The updates are written into the data, so they share its element type.
func.func @updates_element_mismatch(%ctx: !hipsr.context,
                                    %data: tensor<4x2xf16>,
                                    %ids: tensor<5x2xi64>,
                                    %updates: tensor<5xf32>)
    -> tensor<4x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.ScatterND'}}
  %0 = "onnx.ScatterND"(%data, %ids, %updates) {reduction = "none"}
      : (tensor<4x2xf16>, tensor<5x2xi64>, tensor<5xf32>) -> tensor<4x2xf16>
  return %0 : tensor<4x2xf16>
}

// -----

// The updates shape follows from all three ranks, so all three must be ranked.
func.func @unranked_updates(%ctx: !hipsr.context, %data: tensor<4x2xf16>,
                            %ids: tensor<5x2xi64>,
                            %updates: tensor<*xf16>) -> tensor<4x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.ScatterND'}}
  %0 = "onnx.ScatterND"(%data, %ids, %updates) {reduction = "none"}
      : (tensor<4x2xf16>, tensor<5x2xi64>, tensor<*xf16>) -> tensor<4x2xf16>
  return %0 : tensor<4x2xf16>
}

// -----

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%data: tensor<4x2xf16>, %ids: tensor<5x2xi64>,
                           %updates: tensor<5xf16>) -> tensor<4x2xf16> {
  // expected-error @+1 {{failed to legalize operation 'onnx.ScatterND'}}
  %0 = "onnx.ScatterND"(%data, %ids, %updates) {reduction = "none"}
      : (tensor<4x2xf16>, tensor<5x2xi64>, tensor<5xf16>) -> tensor<4x2xf16>
  return %0 : tensor<4x2xf16>
}
