// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }

  // ---- ReduceMean over last dim ----
  func.func @test_reduce_mean(%arg0: tensor<2x4x16xf32>) -> tensor<2x4x1xf32> {
    %axes = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.ReduceMean"(%arg0, %axes) {keepdims = 1 : si64}
        : (tensor<2x4x16xf32>, tensor<1xi64>) -> tensor<2x4x1xf32>
    return %0 : tensor<2x4x1xf32>
  }
  // CHECK-LABEL: func.func @test_reduce_mean
  // CHECK: hip.reduce_mean(%{{.*}})

  // ---- Concat along axis 1 ----
  func.func @test_concat(%a: tensor<2x4xf16>, %b: tensor<2x6xf16>) -> tensor<2x10xf16> {
    %0 = "onnx.Concat"(%a, %b) {axis = 1 : si64}
        : (tensor<2x4xf16>, tensor<2x6xf16>) -> tensor<2x10xf16>
    return %0 : tensor<2x10xf16>
  }
  // CHECK-LABEL: func.func @test_concat
  // CHECK: hip.concat(%{{.*}})
  // CHECK-SAME: axis = 1

  // ---- ConstantOfShape (default fill = 0.0) ----
  func.func @test_constant_of_shape(%arg0: tensor<2xi64>) -> tensor<3x4xf32> {
    %0 = "onnx.ConstantOfShape"(%arg0)
        {value = dense<7.0> : tensor<1xf32>}
        : (tensor<2xi64>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
  // CHECK-LABEL: func.func @test_constant_of_shape
  // CHECK: hip.constant_of_shape(%{{.*}})

  // ---- Shape (folded to arith.constant) ----
  func.func @test_shape(%arg0: tensor<2x4x16xf32>) -> tensor<3xi64> {
    %0 = "onnx.Shape"(%arg0) : (tensor<2x4x16xf32>) -> tensor<3xi64>
    return %0 : tensor<3xi64>
  }
  // CHECK-LABEL: func.func @test_shape
  // CHECK: arith.constant dense<[2, 4, 16]>

  // ---- Range (folded to arith.constant) ----
  func.func @test_range() -> tensor<5xi64> {
    %s = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %l = "onnx.Constant"() {value = dense<5> : tensor<i64>} : () -> tensor<i64>
    %d = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %0 = "onnx.Range"(%s, %l, %d) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<5xi64>
    return %0 : tensor<5xi64>
  }
  // CHECK-LABEL: func.func @test_range
  // CHECK: arith.constant dense<[0, 1, 2, 3, 4]>

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
