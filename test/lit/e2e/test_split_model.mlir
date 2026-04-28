// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s
//
// Verifies ONNX Split lowers through hip.split and LLVM wrap_split.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 2
// CHECK: llvm.func @wrap_split
// CHECK-NOT: onnx.Split
module {
  func.func @main_graph(%arg0: tensor<1x4xf32> {onnx.name = "input_0"})
      -> (tensor<1x2xf32> {onnx.name = "output_0"}, tensor<1x2xf32> {onnx.name = "output_1"}) {
    %out0, %out1 = "onnx.Split"(%arg0) {axis = 1 : si64, onnx_node_name = "Split_0"}
      : (tensor<1x4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
    "onnx.Return"(%out0, %out1) : (tensor<1x2xf32>, tensor<1x2xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
