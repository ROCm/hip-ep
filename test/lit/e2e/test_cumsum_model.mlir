// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_cumsum
// CHECK-NOT: onnx.CumSum

module {
  func.func @main_graph(%arg0: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
    %axis = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %0 = "onnx.CumSum"(%arg0, %axis) {exclusive = 0 : si64, reverse = 0 : si64}
        : (tensor<2x4xf32>, tensor<i64>) -> tensor<2x4xf32>
    "onnx.Return"(%0) : (tensor<2x4xf32>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
