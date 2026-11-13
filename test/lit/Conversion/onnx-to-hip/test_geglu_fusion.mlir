// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify that the GEGLU feed-forward chain (value / gate Slices of one
// projection, exact GELU of the gate as primitives, gating Mul) folds into a
// single hip.geglu over the projection, in the spellings exporters and the
// pre-lowering rewrites produce, and that a gate taken from a different
// projection does not.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1xf16>) -> tensor<1x1xf16> {
    return %arg0 : tensor<1x1xf16>
  }

  // diffusers export: Mul(Mul(g, Add(Erf(Div(g, sqrt2)), 1)), 0.5), then
  // Mul(h, gelu).
  // CHECK-LABEL: func.func @geglu_diffusers
  // CHECK-NOT: hip.erf
  // CHECK: hip.geglu(%{{.*}}) ins(%{{.*}} : tensor<2x16x64xf16>) outs(%{{.*}} : tensor<2x16x32xf16>)
  // CHECK-NOT: hip.erf
  // CHECK-NOT: hip.mul
  // CHECK: return
  func.func @geglu_diffusers(%p: tensor<2x16x64xf16>) -> tensor<2x16x32xf16> {
    %c0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %c32 = "onnx.Constant"() {value = dense<32> : tensor<1xi64>} : () -> tensor<1xi64>
    %c64 = "onnx.Constant"() {value = dense<64> : tensor<1xi64>} : () -> tensor<1xi64>
    %axis = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %sqrt2 = "onnx.Constant"() {value = dense<1.4140625> : tensor<f16>} : () -> tensor<f16>
    %one = "onnx.Constant"() {value = dense<1.0> : tensor<f16>} : () -> tensor<f16>
    %half = "onnx.Constant"() {value = dense<0.5> : tensor<f16>} : () -> tensor<f16>
    %h = "onnx.Slice"(%p, %c0, %c32, %axis) : (tensor<2x16x64xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x16x32xf16>
    %g = "onnx.Slice"(%p, %c32, %c64, %axis) : (tensor<2x16x64xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x16x32xf16>
    %d = "onnx.Div"(%g, %sqrt2) : (tensor<2x16x32xf16>, tensor<f16>) -> tensor<2x16x32xf16>
    %e = "onnx.Erf"(%d) : (tensor<2x16x32xf16>) -> tensor<2x16x32xf16>
    %a = "onnx.Add"(%e, %one) : (tensor<2x16x32xf16>, tensor<f16>) -> tensor<2x16x32xf16>
    %m = "onnx.Mul"(%g, %a) : (tensor<2x16x32xf16>, tensor<2x16x32xf16>) -> tensor<2x16x32xf16>
    %gelu = "onnx.Mul"(%m, %half) : (tensor<2x16x32xf16>, tensor<f16>) -> tensor<2x16x32xf16>
    %y = "onnx.Mul"(%h, %gelu) : (tensor<2x16x32xf16>, tensor<2x16x32xf16>) -> tensor<2x16x32xf16>
    return %y : tensor<2x16x32xf16>
  }

  // The Div already rewritten to Mul(Reciprocal), the 0.5 applied to g first,
  // and the gate multiplied from the left.
  // CHECK-LABEL: func.func @geglu_reciprocal
  // CHECK: hip.geglu(%{{.*}}) ins(%{{.*}} : tensor<8x64xf16>) outs(%{{.*}} : tensor<8x32xf16>)
  // CHECK-NOT: hip.erf
  // CHECK: return
  func.func @geglu_reciprocal(%p: tensor<8x64xf16>) -> tensor<8x32xf16> {
    %c0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %c32 = "onnx.Constant"() {value = dense<32> : tensor<1xi64>} : () -> tensor<1xi64>
    %end = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
    %axis = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %sqrt2 = "onnx.Constant"() {value = dense<1.4140625> : tensor<1xf16>} : () -> tensor<1xf16>
    %one = "onnx.Constant"() {value = dense<1.0> : tensor<1xf16>} : () -> tensor<1xf16>
    %half = "onnx.Constant"() {value = dense<0.5> : tensor<1xf16>} : () -> tensor<1xf16>
    %h = "onnx.Slice"(%p, %c0, %c32, %axis) : (tensor<8x64xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<8x32xf16>
    %g = "onnx.Slice"(%p, %c32, %end, %axis) : (tensor<8x64xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<8x32xf16>
    %r = "onnx.Reciprocal"(%sqrt2) : (tensor<1xf16>) -> tensor<1xf16>
    %d = "onnx.Mul"(%g, %r) : (tensor<8x32xf16>, tensor<1xf16>) -> tensor<8x32xf16>
    %e = "onnx.Erf"(%d) : (tensor<8x32xf16>) -> tensor<8x32xf16>
    %a = "onnx.Add"(%one, %e) : (tensor<1xf16>, tensor<8x32xf16>) -> tensor<8x32xf16>
    %gh = "onnx.Mul"(%g, %half) : (tensor<8x32xf16>, tensor<1xf16>) -> tensor<8x32xf16>
    %gelu = "onnx.Mul"(%gh, %a) : (tensor<8x32xf16>, tensor<8x32xf16>) -> tensor<8x32xf16>
    %y = "onnx.Mul"(%gelu, %h) : (tensor<8x32xf16>, tensor<8x32xf16>) -> tensor<8x32xf16>
    return %y : tensor<8x32xf16>
  }

  // Value and gate from different tensors: not GEGLU.
  // CHECK-LABEL: func.func @two_projections
  // CHECK-NOT: hip.geglu
  // CHECK: return
  func.func @two_projections(%p: tensor<8x64xf16>, %q: tensor<8x64xf16>) -> tensor<8x32xf16> {
    %c0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %c32 = "onnx.Constant"() {value = dense<32> : tensor<1xi64>} : () -> tensor<1xi64>
    %c64 = "onnx.Constant"() {value = dense<64> : tensor<1xi64>} : () -> tensor<1xi64>
    %axis = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %sqrt2 = "onnx.Constant"() {value = dense<1.4140625> : tensor<f16>} : () -> tensor<f16>
    %one = "onnx.Constant"() {value = dense<1.0> : tensor<f16>} : () -> tensor<f16>
    %half = "onnx.Constant"() {value = dense<0.5> : tensor<f16>} : () -> tensor<f16>
    %h = "onnx.Slice"(%p, %c0, %c32, %axis) : (tensor<8x64xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<8x32xf16>
    %g = "onnx.Slice"(%q, %c32, %c64, %axis) : (tensor<8x64xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<8x32xf16>
    %d = "onnx.Div"(%g, %sqrt2) : (tensor<8x32xf16>, tensor<f16>) -> tensor<8x32xf16>
    %e = "onnx.Erf"(%d) : (tensor<8x32xf16>) -> tensor<8x32xf16>
    %a = "onnx.Add"(%e, %one) : (tensor<8x32xf16>, tensor<f16>) -> tensor<8x32xf16>
    %m = "onnx.Mul"(%g, %a) : (tensor<8x32xf16>, tensor<8x32xf16>) -> tensor<8x32xf16>
    %gelu = "onnx.Mul"(%m, %half) : (tensor<8x32xf16>, tensor<f16>) -> tensor<8x32xf16>
    %y = "onnx.Mul"(%h, %gelu) : (tensor<8x32xf16>, tensor<8x32xf16>) -> tensor<8x32xf16>
    return %y : tensor<8x32xf16>
  }
}
