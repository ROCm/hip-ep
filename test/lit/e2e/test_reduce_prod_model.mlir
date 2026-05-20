// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_reduce_prod
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.ReduceProd
module {
  func.func @main_graph(%arg0: tensor<3x2x2xf32> {onnx.name = "data"}) -> (tensor<1x1x1xf32> {onnx.name = "output"}) {
    %0 = "onnx.ReduceProd"(%arg0) {keepdims = 1 : si64, onnx_node_name = "reduce_prod_node"} : (tensor<3x2x2xf32>) -> tensor<1x1x1xf32>
    "onnx.Return"(%0) : (tensor<1x1x1xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
