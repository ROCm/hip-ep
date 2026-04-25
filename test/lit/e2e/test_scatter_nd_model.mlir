// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_scatter_nd
// CHECK-NOT: onnx.ScatterND

module {
  func.func @main_graph(%data: tensor<8x4xf32>, %updates: tensor<3x4xf32>)
      -> tensor<8x4xf32> {
    %indices = "onnx.Constant"() {value = dense<[[0],[2],[5]]> : tensor<3x1xi64>}
        : () -> tensor<3x1xi64>
    %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"}
        : (tensor<8x4xf32>, tensor<3x1xi64>, tensor<3x4xf32>) -> tensor<8x4xf32>
    "onnx.Return"(%0) : (tensor<8x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
