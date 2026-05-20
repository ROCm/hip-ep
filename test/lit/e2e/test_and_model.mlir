// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_and
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.And
module {
  func.func @main_graph(%arg0: tensor<3x4xi1> {onnx.name = "a"}, %arg1: tensor<3x4xi1> {onnx.name = "b"}) -> (tensor<3x4xi1> {onnx.name = "output"}) {
    %0 = "onnx.And"(%arg0, %arg1) {onnx_node_name = "and_node"} : (tensor<3x4xi1>, tensor<3x4xi1>) -> tensor<3x4xi1>
    "onnx.Return"(%0) : (tensor<3x4xi1>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
