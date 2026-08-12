// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.Slice to hipsr.slice. Rejected forms live in
// slice-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The window operands become attributes, so only the data reaches the
// placeholder. Its shape region stays empty: hipsr.slice is DPS, so
// hipsr-populate-shape-region has a recipe to dispatch on.
// CHECK-LABEL: func.func @strided_window(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context, %[[DATA:.*]]: tensor<8x4xf16>) -> tensor<3x4xf16> {
// CHECK:         %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[DATA]] : tensor<8x4xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<3x4xf16>
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.slice(%[[CTX]]) ins(%[[DATA]] : tensor<8x4xf16>) outs(%[[INIT]] : tensor<3x4xf16>) {axes = array<i64: 0>, ends = array<i64: 7>, starts = array<i64: 1>, steps = array<i64: 2>} : tensor<3x4xf16>
// CHECK-NEXT:    return %[[RESULT]] : tensor<3x4xf16>
func.func @strided_window(%ctx: !hipsr.context,
                          %data: tensor<8x4xf16>) -> tensor<3x4xf16> {
  %starts = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends = "onnx.Constant"() {value = dense<7> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %steps = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes, %steps)
      : (tensor<8x4xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
         tensor<1xi64>) -> tensor<3x4xf16>
  return %0 : tensor<3x4xf16>
}

// -----

// Without the optional operands, ONNX slices every axis at unit stride; the
// hipsr operation always names both, so the conversion materializes them.
// CHECK-LABEL: func.func @default_axes_and_steps(
// CHECK:         hipsr.slice(%{{.*}}) ins(%{{.*}} : tensor<4x6xf16>) outs(%{{.*}} : tensor<2x3xf16>) {axes = array<i64: 0, 1>, ends = array<i64: 3, 5>, starts = array<i64: 1, 2>, steps = array<i64: 1, 1>} : tensor<2x3xf16>
func.func @default_axes_and_steps(%ctx: !hipsr.context,
                                  %data: tensor<4x6xf16>) -> tensor<2x3xf16> {
  %starts = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[3, 5]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0 = "onnx.Slice"(%data, %starts, %ends)
      : (tensor<4x6xf16>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// A negative bound counts back from the end, a bound past the end saturates
// (which is how ONNX spells "to the end"), and a negative axis normalizes.
// CHECK-LABEL: func.func @relative_bounds(
// CHECK:         hipsr.slice(%{{.*}}) ins(%{{.*}} : tensor<8xf16>) outs(%{{.*}} : tensor<2xf16>) {axes = array<i64: 0>, ends = array<i64: 8>, starts = array<i64: 6>, steps = array<i64: 1>} : tensor<2xf16>
func.func @relative_bounds(%ctx: !hipsr.context,
                           %data: tensor<8xf16>) -> tensor<2xf16> {
  %starts = "onnx.Constant"() {value = dense<-2> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes)
      : (tensor<8xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<2xf16>
  return %0 : tensor<2xf16>
}

// -----

// An untouched axis passes through, so it may stay dynamic.
// CHECK-LABEL: func.func @untouched_dynamic_axis(
// CHECK:         hipsr.slice(%{{.*}}) ins(%{{.*}} : tensor<?x6xf16>) outs(%{{.*}} : tensor<?x3xf16>) {axes = array<i64: 1>, ends = array<i64: 5>, starts = array<i64: 2>, steps = array<i64: 1>} : tensor<?x3xf16>
func.func @untouched_dynamic_axis(%ctx: !hipsr.context,
                                  %data: tensor<?x6xf16>) -> tensor<?x3xf16> {
  %starts = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends = "onnx.Constant"() {value = dense<5> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes)
      : (tensor<?x6xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<?x3xf16>
  return %0 : tensor<?x3xf16>
}
