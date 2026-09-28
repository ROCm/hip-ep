// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-tosa %s | FileCheck %s

// ONNX Gather along an inner axis is flattened to TOSA's batched
// [N,K,C] x [N,W] form. The compare/add/select sequence normalizes ONNX's
// negative indices before the gather.
// CHECK-LABEL: func.func @gather_inner_axis
// CHECK: tosa.cast
// CHECK: tosa.tile
// CHECK: tosa.greater
// CHECK: tosa.add
// CHECK: tosa.select
// CHECK: tosa.gather
// CHECK-NOT: hip.gather
func.func @gather_inner_axis(
    %ctx: !hip.context, %data: tensor<2x3x4xf16>,
    %indices: tensor<2xi64>, %init: tensor<2x2x4xf16>)
    -> tensor<2x2x4xf16> attributes {rock.kernel} {
  %result = hip.gather(%ctx)
      ins(%data, %indices : tensor<2x3x4xf16>, tensor<2xi64>)
      outs(%init : tensor<2x2x4xf16>) {axis = 1 : i64}
      : tensor<2x2x4xf16>
  return %result : tensor<2x2x4xf16>
}

// Arbitrary-rank indices replace the gathered axis in the ONNX output shape.
// CHECK-LABEL: func.func @gather_negative_axis
// CHECK: tosa.gather
// CHECK: tosa.reshape
// CHECK-NOT: hip.gather
func.func @gather_negative_axis(
    %ctx: !hip.context, %data: tensor<2x3xf32>,
    %indices: tensor<2x2xi32>, %init: tensor<2x2x2xf32>)
    -> tensor<2x2x2xf32> attributes {rock.kernel} {
  %result = hip.gather(%ctx)
      ins(%data, %indices : tensor<2x3xf32>, tensor<2x2xi32>)
      outs(%init : tensor<2x2x2xf32>) {axis = -1 : i64}
      : tensor<2x2x2xf32>
  return %result : tensor<2x2x2xf32>
}
