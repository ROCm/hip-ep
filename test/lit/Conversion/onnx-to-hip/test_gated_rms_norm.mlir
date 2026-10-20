// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf16>) -> tensor<4xf16> {
    return %arg0 : tensor<4xf16>
  }

  // CHECK-LABEL: func.func @gated_rms_norm_sigmoid
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x?x6144xf16>, %[[S:.*]]: tensor<128xf16>, %[[Z:.*]]: tensor<?x?x6144xf16>)
  // CHECK-NOT: onnx.Custom
  // CHECK: hip.gated_rms_norm(%[[CTX]]) ins(%[[X]], %[[S]], %[[Z]] : tensor<?x?x6144xf16>, tensor<128xf16>, tensor<?x?x6144xf16>) outs({{.*}} : tensor<?x?x6144xf16>) {activation = "sigmoid", epsilon = 9.99999997E-7 : f32}
  func.func @gated_rms_norm_sigmoid(%x: tensor<?x?x6144xf16>, %s: tensor<128xf16>, %z: tensor<?x?x6144xf16>) -> tensor<?x?x6144xf16> {
    %y = "onnx.Custom"(%x, %s, %z) {
      function_name = "GatedRMSNorm",
      domain_name = "com.microsoft",
      activation = "sigmoid",
      epsilon = 9.99999997E-7 : f32
    } : (tensor<?x?x6144xf16>, tensor<128xf16>, tensor<?x?x6144xf16>) -> tensor<?x?x6144xf16>
    return %y : tensor<?x?x6144xf16>
  }

  // Defaults: activation silu ("swish" is its alias), epsilon 1e-5.
  // CHECK-LABEL: func.func @gated_rms_norm_defaults
  // CHECK: hip.gated_rms_norm
  // CHECK-SAME: {activation = "silu", epsilon = 9.99999974E-6 : f32}
  func.func @gated_rms_norm_defaults(%x: tensor<2x64xf32>, %s: tensor<64xf32>, %z: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %y = "onnx.Custom"(%x, %s, %z) {
      function_name = "GatedRMSNorm",
      domain_name = "com.microsoft"
    } : (tensor<2x64xf32>, tensor<64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
    return %y : tensor<2x64xf32>
  }
}
