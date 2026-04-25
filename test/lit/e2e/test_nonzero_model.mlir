// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_nonzero
// CHECK-NOT: onnx.NonZero

module {
  func.func @main_graph(%arg0: tensor<8xi64>) -> tensor<1x8xi64> {
    %0 = "onnx.NonZero"(%arg0) : (tensor<8xi64>) -> tensor<1x8xi64>
    "onnx.Return"(%0) : (tensor<1x8xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
