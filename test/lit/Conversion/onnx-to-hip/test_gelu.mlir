// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128x768xf16>) -> tensor<1x128x768xf16> {
    return %arg0 : tensor<1x128x768xf16>
  }

  // Test 1: Basic GELU with f32
  // CHECK-LABEL: func.func @test_gelu_f32
  // CHECK-NOT: onnx.Gelu
  // CHECK: hip.gelu
  func.func @test_gelu_f32(%input: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
    %output = "onnx.Gelu"(%input) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    return %output : tensor<1x1024xf32>
  }

  // Test 2: GELU with f16
  // CHECK-LABEL: func.func @test_gelu_f16
  // CHECK-NOT: onnx.Gelu
  // CHECK: hip.gelu
  func.func @test_gelu_f16(%input: tensor<1x128x768xf16>) -> tensor<1x128x768xf16> {
    %output = "onnx.Gelu"(%input) : (tensor<1x128x768xf16>) -> tensor<1x128x768xf16>
    return %output : tensor<1x128x768xf16>
  }

  // Test 3: GELU with bf16
  // CHECK-LABEL: func.func @test_gelu_bf16
  // CHECK-NOT: onnx.Gelu
  // CHECK: hip.gelu
  func.func @test_gelu_bf16(%input: tensor<2x512xbf16>) -> tensor<2x512xbf16> {
    %output = "onnx.Gelu"(%input) : (tensor<2x512xbf16>) -> tensor<2x512xbf16>
    return %output : tensor<2x512xbf16>
  }

  // Test 4: GELU with dynamic shape
  // CHECK-LABEL: func.func @test_gelu_dynamic
  // CHECK-NOT: onnx.Gelu
  // CHECK: hip.gelu
  func.func @test_gelu_dynamic(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %output = "onnx.Gelu"(%input) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %output : tensor<?x?xf32>
  }

  // Test 5: GELU with 4D tensor (batch, seq_len, hidden_dim)
  // CHECK-LABEL: func.func @test_gelu_4d
  // CHECK-NOT: onnx.Gelu
  // CHECK: hip.gelu
  func.func @test_gelu_4d(%input: tensor<2x128x12x64xf16>) -> tensor<2x128x12x64xf16> {
    %output = "onnx.Gelu"(%input) : (tensor<2x128x12x64xf16>) -> tensor<2x128x12x64xf16>
    return %output : tensor<2x128x12x64xf16>
  }

  // Test 6: GELU with 1D tensor
  // CHECK-LABEL: func.func @test_gelu_1d
  // CHECK-NOT: onnx.Gelu
  // CHECK: hip.gelu
  func.func @test_gelu_1d(%input: tensor<768xf32>) -> tensor<768xf32> {
    %output = "onnx.Gelu"(%input) : (tensor<768xf32>) -> tensor<768xf32>
    return %output : tensor<768xf32>
  }

  // Test 7: GELU with approximate="tanh"
  // CHECK-LABEL: func.func @test_gelu_approximate_tanh
  // CHECK-NOT: onnx.Gelu
  // CHECK: hip.gelu{{.*}}approximate = "tanh"
  func.func @test_gelu_approximate_tanh(%input: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
    %output = "onnx.Gelu"(%input) {approximate = "tanh"} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    return %output : tensor<1x1024xf32>
  }

  // Test 8: GELU with approximate="none" (exact erf, default)
  // CHECK-LABEL: func.func @test_gelu_approximate_none
  // CHECK-NOT: onnx.Gelu
  // CHECK: hip.gelu
  // CHECK-NOT: approximate
  func.func @test_gelu_approximate_none(%input: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
    %output = "onnx.Gelu"(%input) {approximate = "none"} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    return %output : tensor<1x1024xf32>
  }

  // Test 9: GELU with f64 (double) - ONNX spec compliance
  // CHECK-LABEL: func.func @test_gelu_f64
  // CHECK-NOT: onnx.Gelu
  // CHECK: hip.gelu
  func.func @test_gelu_f64(%input: tensor<1x1024xf64>) -> tensor<1x1024xf64> {
    %output = "onnx.Gelu"(%input) : (tensor<1x1024xf64>) -> tensor<1x1024xf64>
    return %output : tensor<1x1024xf64>
  }
}
