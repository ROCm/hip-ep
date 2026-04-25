// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }

  // ---- Sin (kind = 0) ----
  func.func @test_sin(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %0 = "onnx.Sin"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  // CHECK-LABEL: func.func @test_sin
  // CHECK: hip.unary_elementwise(%{{.*}})
  // CHECK-SAME: kind = 0

  // ---- Cos (kind = 1) ----
  func.func @test_cos(%arg0: tensor<128xf16>) -> tensor<128xf16> {
    %0 = "onnx.Cos"(%arg0) : (tensor<128xf16>) -> tensor<128xf16>
    return %0 : tensor<128xf16>
  }
  // CHECK-LABEL: func.func @test_cos
  // CHECK: hip.unary_elementwise(%{{.*}})
  // CHECK-SAME: kind = 1

  // ---- Exp (kind = 2) ----
  func.func @test_exp(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = "onnx.Exp"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  // CHECK-LABEL: func.func @test_exp
  // CHECK: hip.unary_elementwise(%{{.*}})
  // CHECK-SAME: kind = 2

  // ---- Tanh (kind = 3) ----
  func.func @test_tanh(%arg0: tensor<1x128x512xf16>) -> tensor<1x128x512xf16> {
    %0 = "onnx.Tanh"(%arg0) : (tensor<1x128x512xf16>) -> tensor<1x128x512xf16>
    return %0 : tensor<1x128x512xf16>
  }
  // CHECK-LABEL: func.func @test_tanh
  // CHECK: hip.unary_elementwise(%{{.*}})
  // CHECK-SAME: kind = 3

  // ---- Floor (kind = 4) ----
  func.func @test_floor(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %0 = "onnx.Floor"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  // CHECK-LABEL: func.func @test_floor
  // CHECK: hip.unary_elementwise(%{{.*}})
  // CHECK-SAME: kind = 4

  // ---- Round (kind = 5) ----
  func.func @test_round(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %0 = "onnx.Round"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  // CHECK-LABEL: func.func @test_round
  // CHECK: hip.unary_elementwise(%{{.*}})
  // CHECK-SAME: kind = 5

  // ---- Atan (kind = 6) ----
  func.func @test_atan(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %0 = "onnx.Atan"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  // CHECK-LABEL: func.func @test_atan
  // CHECK: hip.unary_elementwise(%{{.*}})
  // CHECK-SAME: kind = 6

  // ---- LeakyRelu (kind = 7) ----
  // Default alpha is 0.01 if not supplied.  We pass it explicitly here to
  // keep the FileCheck deterministic.
  func.func @test_leaky_relu(%arg0: tensor<1x128x512xf16>) -> tensor<1x128x512xf16> {
    %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.125 : f32}
        : (tensor<1x128x512xf16>) -> tensor<1x128x512xf16>
    return %0 : tensor<1x128x512xf16>
  }
  // CHECK-LABEL: func.func @test_leaky_relu
  // CHECK: hip.unary_elementwise(%{{.*}})
  // CHECK-SAME: alpha = 1.250000e-01
  // CHECK-SAME: kind = 7

  // ---- Clip (kind = 8) ----  min = -1, max = 6 (clip to [min, max])
  func.func @test_clip(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %min = "onnx.Constant"() {value = dense<-1.0> : tensor<f32>} : () -> tensor<f32>
    %max = "onnx.Constant"() {value = dense<6.0> : tensor<f32>}  : () -> tensor<f32>
    %0 = "onnx.Clip"(%arg0, %min, %max)
        : (tensor<128xf32>, tensor<f32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  // CHECK-LABEL: func.func @test_clip
  // CHECK: hip.unary_elementwise(%{{.*}})
  // CHECK-SAME: alpha = -1.000000e+00
  // CHECK-SAME: beta = 6.000000e+00
  // CHECK-SAME: kind = 8

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
