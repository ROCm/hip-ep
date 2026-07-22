// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_gather_elements
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.GatherElements
module {
  func.func @main_graph(%arg0: tensor<2x2xf32> {onnx.name = "data"},
                          %arg1: tensor<2x2xi32> {onnx.name = "indices"})
      -> (tensor<2x2xf32> {onnx.name = "output"}) {
    %0 = "onnx.GatherElements"(%arg0, %arg1) {
      axis = 1 : si64,
      onnx_node_name = "gather_elements_node"
    } : (tensor<2x2xf32>, tensor<2x2xi32>) -> tensor<2x2xf32>
    "onnx.Return"(%0) : (tensor<2x2xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
