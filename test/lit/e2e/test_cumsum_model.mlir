// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_cumsum
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.CumSum
module {
  func.func @main_graph(%arg0: tensor<3x4xf32> {onnx.name = "x"}, %arg1: tensor<i64> {onnx.name = "axis"}) -> (tensor<3x4xf32> {onnx.name = "output"}) {
    %0 = "onnx.CumSum"(%arg0, %arg1) {onnx_node_name = "cumsum_node", exclusive = 0 : si64, reverse = 0 : si64} : (tensor<3x4xf32>, tensor<i64>) -> tensor<3x4xf32>
    "onnx.Return"(%0) : (tensor<3x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
