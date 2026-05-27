// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_gather_nd
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.GatherND
module {
  func.func @main_graph(%arg0: tensor<2x2xf32> {onnx.name = "data"}, %arg1: tensor<2x2xi64> {onnx.name = "indices"}) -> (tensor<2xf32> {onnx.name = "output"}) {
    %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64, onnx_node_name = "gather_nd_node"} : (tensor<2x2xf32>, tensor<2x2xi64>) -> tensor<2xf32>
    "onnx.Return"(%0) : (tensor<2xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
