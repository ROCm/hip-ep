// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }

  // ---- CumSum along axis 1, inclusive, forward ----
  func.func @test_cumsum(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %axis = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %0 = "onnx.CumSum"(%arg0, %axis) {exclusive = 0 : si64, reverse = 0 : si64}
        : (tensor<2x4xf32>, tensor<i64>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  // CHECK-LABEL: func.func @test_cumsum
  // CHECK: hip.cumsum(%{{.*}})
  // CHECK-SAME: axis = 1

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
