// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the pre-lowering fold of Cast(f16->f32) on skip/gamma and
// Cast(f32->f16) on the output into an fp32 SkipSimplifiedLayerNormalization
// (the gpt-oss residual layout): hip.skip_rms_norm reads the f16 skip and
// gamma, writes the f16 output and keeps the residual sum in f32, with no
// hip.cast left. Any partial match stays unfused.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1xf16>) -> tensor<1x1xf16> {
    return %arg0 : tensor<1x1xf16>
  }

  func.func @fold(%res: tensor<1x1x2880xf32>, %skip: tensor<1x1x2880xf16>)
      -> (tensor<1x1x2880xf16>, tensor<1x1x2880xf32>) {
    %g16 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2880xf16>} : () -> tensor<2880xf16>
    %s32 = "onnx.Cast"(%skip) {to = 1 : si64} : (tensor<1x1x2880xf16>) -> tensor<1x1x2880xf32>
    %g32 = "onnx.Cast"(%g16) {to = 1 : si64} : (tensor<2880xf16>) -> tensor<2880xf32>
    %0:4 = "onnx.Custom"(%res, %s32, %g32) {
      function_name = "SkipSimplifiedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<1x1x2880xf32>, tensor<1x1x2880xf32>, tensor<2880xf32>)
        -> (tensor<1x1x2880xf32>, none, none, tensor<1x1x2880xf32>)
    %y16 = "onnx.Cast"(%0#0) {to = 10 : si64} : (tensor<1x1x2880xf32>) -> tensor<1x1x2880xf16>
    return %y16, %0#3 : tensor<1x1x2880xf16>, tensor<1x1x2880xf32>
  }

  // CHECK-LABEL: func.func @fold
  // CHECK-NOT: hip.cast
  // CHECK: hip.skip_rms_norm(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x1x2880xf32>, tensor<1x1x2880xf16>, tensor<2880xf16>)
  // CHECK-SAME: outs(%{{.*}}, %{{.*}} : tensor<1x1x2880xf16>, tensor<1x1x2880xf32>)
  // CHECK-NOT: hip.cast
  // CHECK: return

  // The f32 output is also returned, so the output Cast cannot be absorbed.
  func.func @output_shared(%res: tensor<1x1x2880xf32>, %skip: tensor<1x1x2880xf16>)
      -> (tensor<1x1x2880xf16>, tensor<1x1x2880xf32>) {
    %g16 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2880xf16>} : () -> tensor<2880xf16>
    %s32 = "onnx.Cast"(%skip) {to = 1 : si64} : (tensor<1x1x2880xf16>) -> tensor<1x1x2880xf32>
    %g32 = "onnx.Cast"(%g16) {to = 1 : si64} : (tensor<2880xf16>) -> tensor<2880xf32>
    %0:4 = "onnx.Custom"(%res, %s32, %g32) {
      function_name = "SkipSimplifiedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<1x1x2880xf32>, tensor<1x1x2880xf32>, tensor<2880xf32>)
        -> (tensor<1x1x2880xf32>, none, none, tensor<1x1x2880xf32>)
    %y16 = "onnx.Cast"(%0#0) {to = 10 : si64} : (tensor<1x1x2880xf32>) -> tensor<1x1x2880xf16>
    return %y16, %0#0 : tensor<1x1x2880xf16>, tensor<1x1x2880xf32>
  }

  // CHECK-LABEL: func.func @output_shared
  // CHECK: hip.cast
  // CHECK: hip.skip_rms_norm(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x1x2880xf32>, tensor<1x1x2880xf32>, tensor<2880xf32>)
  // CHECK-SAME: outs(%{{.*}}, %{{.*}} : tensor<1x1x2880xf32>, tensor<1x1x2880xf32>)
  // CHECK: hip.cast
  // CHECK: return

  // Gamma is already f32: nothing to peel, so the fold does not fire.
  func.func @gamma_f32(%res: tensor<1x1x2880xf32>, %skip: tensor<1x1x2880xf16>)
      -> tensor<1x1x2880xf16> {
    %g32 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2880xf32>} : () -> tensor<2880xf32>
    %s32 = "onnx.Cast"(%skip) {to = 1 : si64} : (tensor<1x1x2880xf16>) -> tensor<1x1x2880xf32>
    %0:4 = "onnx.Custom"(%res, %s32, %g32) {
      function_name = "SkipSimplifiedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<1x1x2880xf32>, tensor<1x1x2880xf32>, tensor<2880xf32>)
        -> (tensor<1x1x2880xf32>, none, none, tensor<1x1x2880xf32>)
    %y16 = "onnx.Cast"(%0#0) {to = 10 : si64} : (tensor<1x1x2880xf32>) -> tensor<1x1x2880xf16>
    return %y16 : tensor<1x1x2880xf16>
  }

  // CHECK-LABEL: func.func @gamma_f32
  // CHECK: hip.skip_rms_norm(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x1x2880xf32>, tensor<1x1x2880xf32>, tensor<2880xf32>)
  // CHECK: hip.cast
  // CHECK: return

  // A bias input keeps the all-f32 form.
  func.func @with_bias(%res: tensor<1x1x2880xf32>, %skip: tensor<1x1x2880xf16>,
                       %bias: tensor<2880xf32>) -> tensor<1x1x2880xf16> {
    %g16 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2880xf16>} : () -> tensor<2880xf16>
    %s32 = "onnx.Cast"(%skip) {to = 1 : si64} : (tensor<1x1x2880xf16>) -> tensor<1x1x2880xf32>
    %g32 = "onnx.Cast"(%g16) {to = 1 : si64} : (tensor<2880xf16>) -> tensor<2880xf32>
    %0:4 = "onnx.Custom"(%res, %s32, %g32, %bias) {
      function_name = "SkipSimplifiedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<1x1x2880xf32>, tensor<1x1x2880xf32>, tensor<2880xf32>, tensor<2880xf32>)
        -> (tensor<1x1x2880xf32>, none, none, tensor<1x1x2880xf32>)
    %y16 = "onnx.Cast"(%0#0) {to = 10 : si64} : (tensor<1x1x2880xf32>) -> tensor<1x1x2880xf16>
    return %y16 : tensor<1x1x2880xf16>
  }

  // CHECK-LABEL: func.func @with_bias
  // CHECK: hip.skip_rms_norm(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x1x2880xf32>, tensor<1x1x2880xf32>, tensor<2880xf32>, tensor<2880xf32>)
  // CHECK: hip.cast
  // CHECK: return
}
