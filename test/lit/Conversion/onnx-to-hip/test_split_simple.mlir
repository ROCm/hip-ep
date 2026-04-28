// Minimal test for Split operator
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128x4xf16>) -> tensor<1x128x4xf16> {
    return %arg0 : tensor<1x128x4xf16>
  }

  // Simplest test: equal split into 2 outputs, static shape
  func.func @test_split_simple(%data: tensor<1x4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>) {
    %out0, %out1 = "onnx.Split"(%data) {axis = 1 : si64} : (tensor<1x4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
    return %out0, %out1 : tensor<1x2xf32>, tensor<1x2xf32>
  }
}

// CHECK-LABEL: func.func @test_split_simple
// CHECK-NOT: onnx.Split
// CHECK: hip.split
// CHECK: hip.split
