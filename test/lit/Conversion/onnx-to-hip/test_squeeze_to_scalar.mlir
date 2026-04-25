// RUN: hip-mlir-opt %s --convert-onnx-to-hip | FileCheck %s

// Kokoro emits onnx.Squeeze (tensor<?xi64>) -> tensor<i64> as part of the
// iSTFTNet shape plumbing.  tensor.collapse_shape can't represent
// rank-N -> rank-0 (it requires at least one input dim per output group);
// we lower to tensor.extract from the [0] index instead, trusting that
// the runtime size is 1 (which Kokoro guarantees here).

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: onnx.Squeeze
// CHECK: tensor.extract
// CHECK: tensor.from_elements

module {
  func.func @main_graph(%arg0: !hip.context, %arg1: tensor<?xi64>, %arg2: tensor<1xi64>) -> tensor<i64> {
    %0 = "onnx.Squeeze"(%arg1, %arg2) : (tensor<?xi64>, tensor<1xi64>) -> tensor<i64>
    return %0 : tensor<i64>
  }
}
