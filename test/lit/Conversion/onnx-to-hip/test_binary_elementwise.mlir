// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }

  // ---- Div (kind = 0) ----  same shape on both sides.
  func.func @test_div_same_shape(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> tensor<8x16xf32> {
    %0 = "onnx.Div"(%a, %b) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  // CHECK-LABEL: func.func @test_div_same_shape
  // CHECK: hip.binary_elementwise(%{{.*}})
  // CHECK-SAME: kind = 0

  // ---- Div (kind = 0) ----  rhs scalar broadcast to lhs.
  func.func @test_div_scalar_rhs(%a: tensor<1x128x512xf16>, %b: tensor<f16>)
      -> tensor<1x128x512xf16> {
    %0 = "onnx.Div"(%a, %b)
        : (tensor<1x128x512xf16>, tensor<f16>) -> tensor<1x128x512xf16>
    return %0 : tensor<1x128x512xf16>
  }
  // CHECK-LABEL: func.func @test_div_scalar_rhs
  // CHECK: hip.binary_elementwise(%{{.*}})
  // CHECK-SAME: kind = 0

  // ---- Pow (kind = 1) ----  rhs broadcasts on inner axis.
  func.func @test_pow_broadcast(%a: tensor<8x16xf32>, %b: tensor<16xf32>)
      -> tensor<8x16xf32> {
    %0 = "onnx.Pow"(%a, %b) : (tensor<8x16xf32>, tensor<16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  // CHECK-LABEL: func.func @test_pow_broadcast
  // CHECK: hip.binary_elementwise(%{{.*}})
  // CHECK-SAME: kind = 1

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
