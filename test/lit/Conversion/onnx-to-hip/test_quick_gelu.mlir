// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// com.microsoft QuickGelu is y = x * sigmoid(alpha * x), alpha default 1.702,
// lowered to the existing hip.swish kernel.

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @test_microsoft_quick_gelu
  // CHECK-NOT: onnx.Custom
  // CHECK: hip.swish
  // CHECK-SAME: alpha = 1.702
  func.func @test_microsoft_quick_gelu(%input: tensor<1x77x768xf16>) -> tensor<1x77x768xf16> {
    %output = "onnx.Custom"(%input) {
      function_name = "QuickGelu",
      domain_name = "com.microsoft",
      alpha = 1.702 : f32
    } : (tensor<1x77x768xf16>) -> tensor<1x77x768xf16>
    return %output : tensor<1x77x768xf16>
  }

  // Absent alpha uses the contrib default, which is not hip.swish's default of 1.
  // CHECK-LABEL: func.func @test_microsoft_quick_gelu_default_alpha
  // CHECK-NOT: onnx.Custom
  // CHECK: tensor.dim
  // CHECK: hip.swish
  // CHECK-SAME: alpha = 1.702
  func.func @test_microsoft_quick_gelu_default_alpha(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %output = "onnx.Custom"(%input) {
      function_name = "QuickGelu",
      domain_name = "com.microsoft"
    } : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %output : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @test_microsoft_quick_gelu_alpha
  // CHECK-NOT: onnx.Custom
  // CHECK: hip.swish
  // CHECK-SAME: {alpha = 5.000000e-01 : f64}
  func.func @test_microsoft_quick_gelu_alpha(%input: tensor<4xf32>) -> tensor<4xf32> {
    %output = "onnx.Custom"(%input) {
      function_name = "QuickGelu",
      domain_name = "com.microsoft",
      alpha = 0.5 : f32
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %output : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @test_microsoft_quick_gelu_wrong_domain
  // CHECK: onnx.Custom
  // CHECK-NOT: hip.swish
  func.func @test_microsoft_quick_gelu_wrong_domain(%input: tensor<4xf32>) -> tensor<4xf32> {
    %output = "onnx.Custom"(%input) {
      function_name = "QuickGelu",
      domain_name = "com.example"
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %output : tensor<4xf32>
  }
}
