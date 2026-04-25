// RUN: hip-mlir-opt %s --convert-onnx-to-hip | FileCheck %s

// Verifies that onnx.Shape with a dynamic input dim lowers to a
// tensor.from_elements that mixes static arith.constant ints with a
// tensor.dim op for the dynamic dim, instead of bailing.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: onnx.Shape
// CHECK: tensor.dim
// CHECK: arith.index_cast
// CHECK: tensor.from_elements

module {
  func.func @main_graph(%arg0: !hip.context, %arg1: tensor<?x1xi64>) -> tensor<2xi64> {
    %0 = "onnx.Shape"(%arg1) {start = 0 : si64} : (tensor<?x1xi64>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }
}
