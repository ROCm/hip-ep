// RUN: hip-mlir-opt %s --convert-onnx-to-hip | FileCheck %s

// Kokoro's iSTFTNet noise path emits a degenerate rank-0 onnx.Transpose
// with a perm whose size doesn't match the tensor rank.  We accept it as
// the identity (it's effectively dead code at the model level).

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: onnx.Transpose
// CHECK: return %arg1

module {
  func.func @main_graph(%arg0: !hip.context, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = "onnx.Transpose"(%arg1) {perm = [0, 2, 1, 3]} : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
