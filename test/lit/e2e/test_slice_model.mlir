// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_slice
// CHECK-NOT: onnx.Slice

module {
  func.func @main_graph(%arg0: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
    %starts = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends   = "onnx.Constant"() {value = dense<6> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes   = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes)
        : (tensor<8x16xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4x16xf32>
    "onnx.Return"(%0) : (tensor<4x16xf32>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
