// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_reduce_mean
// CHECK-NOT: onnx.ReduceMean

module {
  func.func @main_graph(%arg0: tensor<2x4x16xf32>) -> (tensor<2x4x1xf32>) {
    %axes = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.ReduceMean"(%arg0, %axes) {keepdims = 1 : si64}
        : (tensor<2x4x16xf32>, tensor<1xi64>) -> tensor<2x4x1xf32>
    "onnx.Return"(%0) : (tensor<2x4x1xf32>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
