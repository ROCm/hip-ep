// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Slice forms the conversion rejects.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file --verify-diagnostics %s

// A runtime bound has no value to resolve against the extent.
func.func @runtime_ends(%ctx: !hipsr.context, %data: tensor<8xf16>,
                        %ends: tensor<1xi64>) -> tensor<?xf16> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Slice'}}
  %0 = "onnx.Slice"(%data, %starts, %ends)
      : (tensor<8xf16>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf16>
  return %0 : tensor<?xf16>
}

// -----

// Neither do runtime axes.
func.func @runtime_axes(%ctx: !hipsr.context, %data: tensor<8xf16>,
                        %axes: tensor<1xi64>) -> tensor<4xf16> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Slice'}}
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes)
      : (tensor<8xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// Reversing an axis is a separate operation.
func.func @negative_step(%ctx: !hipsr.context,
                         %data: tensor<8xf16>) -> tensor<8xf16> {
  %starts = "onnx.Constant"() {value = dense<7> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends = "onnx.Constant"() {value = dense<-9> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %steps = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Slice'}}
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes, %steps)
      : (tensor<8xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
         tensor<1xi64>) -> tensor<8xf16>
  return %0 : tensor<8xf16>
}

// -----

// A bound is resolved against the extent, which a dynamic sliced axis cannot
// supply.
func.func @dynamic_sliced_axis(%ctx: !hipsr.context,
                               %data: tensor<?xf16>) -> tensor<4xf16> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Slice'}}
  %0 = "onnx.Slice"(%data, %starts, %ends)
      : (tensor<?xf16>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// A second window on one axis would overwrite the first.
func.func @repeated_axis(%ctx: !hipsr.context,
                         %data: tensor<8xf16>) -> tensor<2xf16> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<2xi64>} : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[4, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %axes = "onnx.Constant"() {value = dense<0> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Slice'}}
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes)
      : (tensor<8xf16>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>)
      -> tensor<2xf16>
  return %0 : tensor<2xf16>
}

// -----

// Only the data's own axes can be sliced, before or after normalization.
func.func @axis_out_of_range(%ctx: !hipsr.context,
                             %data: tensor<8xf16>) -> tensor<4xf16> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Slice'}}
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes)
      : (tensor<8xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// The four arrays describe the same axes, so their lengths agree.
func.func @length_mismatch(%ctx: !hipsr.context,
                           %data: tensor<8x4xf16>) -> tensor<4x4xf16> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<2xi64>} : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Slice'}}
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes)
      : (tensor<8x4xf16>, tensor<2xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<4x4xf16>
  return %0 : tensor<4x4xf16>
}

// -----

// The stride divides the window, so half of eight at step two is two, not four.
func.func @result_shape_mismatch(%ctx: !hipsr.context,
                                 %data: tensor<8xf16>) -> tensor<4xf16> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %steps = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Slice'}}
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes, %steps)
      : (tensor<8xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
         tensor<1xi64>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// Slice selects elements; it does not convert them.
func.func @element_type_mismatch(%ctx: !hipsr.context,
                                 %data: tensor<8xf16>) -> tensor<4xf32> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Slice'}}
  %0 = "onnx.Slice"(%data, %starts, %ends)
      : (tensor<8xf16>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Every hipsr op threads the context from function argument 0.
func.func @missing_context(%data: tensor<8xf16>) -> tensor<4xf16> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'onnx.Slice'}}
  %0 = "onnx.Slice"(%data, %starts, %ends)
      : (tensor<8xf16>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}
