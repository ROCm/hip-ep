// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_expand_checked
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Expand
module {
  func.func @main_graph(%arg0: tensor<3x1xf32> {onnx.name = "input"}, %arg1: tensor<3xi64> {onnx.name = "shape"}) -> (tensor<2x3x6xf32> {onnx.name = "output"}) {
    %0 = "onnx.Expand"(%arg0, %arg1) {onnx_node_name = "expand_node"} : (tensor<3x1xf32>, tensor<3xi64>) -> tensor<2x3x6xf32>
    "onnx.Return"(%0) : (tensor<2x3x6xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
