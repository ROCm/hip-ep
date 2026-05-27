// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Not E2E full pipeline
// Not: Y = !X, lowered via wrap_not

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_not
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Not
module {
  func.func @main_graph(%arg0: tensor<3x4xi1> {onnx.name = "input"}) -> (tensor<3x4xi1> {onnx.name = "output"}) {
    %0 = "onnx.Not"(%arg0) {onnx_node_name = "not_node"} : (tensor<3x4xi1>) -> tensor<3x4xi1>
    "onnx.Return"(%0) : (tensor<3x4xi1>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
