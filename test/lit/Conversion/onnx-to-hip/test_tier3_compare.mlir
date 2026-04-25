// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }

  // ---- Equal (kind = 0) ----
  func.func @test_equal(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>) -> tensor<8x16xi1> {
    %0 = "onnx.Equal"(%a, %b) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
    return %0 : tensor<8x16xi1>
  }
  // CHECK-LABEL: func.func @test_equal
  // CHECK: hip.compare(%{{.*}})
  // CHECK-SAME: kind = 0

  // ---- Greater (kind = 1) ----
  func.func @test_greater(%a: tensor<8xf16>, %b: tensor<8xf16>) -> tensor<8xi1> {
    %0 = "onnx.Greater"(%a, %b) : (tensor<8xf16>, tensor<8xf16>) -> tensor<8xi1>
    return %0 : tensor<8xi1>
  }
  // CHECK-LABEL: func.func @test_greater
  // CHECK: hip.compare(%{{.*}})
  // CHECK-SAME: kind = 1

  // ---- Less (kind = 2) ----
  func.func @test_less(%a: tensor<8xi64>, %b: tensor<8xi64>) -> tensor<8xi1> {
    %0 = "onnx.Less"(%a, %b) : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
    return %0 : tensor<8xi1>
  }
  // CHECK-LABEL: func.func @test_less
  // CHECK: hip.compare(%{{.*}})
  // CHECK-SAME: kind = 2

  // ---- Where ----
  func.func @test_where(%cond: tensor<4x8xi1>, %x: tensor<4x8xf32>, %y: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %0 = "onnx.Where"(%cond, %x, %y)
        : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
  // CHECK-LABEL: func.func @test_where
  // CHECK: hip.where(%{{.*}})

  // ---- LayerNormalization ----
  func.func @test_layer_norm(%x: tensor<2x4x16xf32>, %gamma: tensor<16xf32>,
                             %beta: tensor<16xf32>) -> tensor<2x4x16xf32> {
    %0 = "onnx.LayerNormalization"(%x, %gamma, %beta)
        {axis = -1 : si64, epsilon = 1.0e-05 : f32, stash_type = 1 : si64}
        : (tensor<2x4x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<2x4x16xf32>
    return %0 : tensor<2x4x16xf32>
  }
  // CHECK-LABEL: func.func @test_layer_norm
  // CHECK: hip.layer_norm(%{{.*}})

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
