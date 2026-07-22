// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_one_hot
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.OneHot
module {
  func.func @main_graph(%arg0: tensor<2x2xi64> {onnx.name = "indices"},
                          %arg1: tensor<i64> {onnx.name = "depth"},
                          %arg2: tensor<2xf32> {onnx.name = "values"})
      -> (tensor<2x10x2xf32> {onnx.name = "output"}) {
    %0 = "onnx.OneHot"(%arg0, %arg1, %arg2) {
      axis = 1 : si64,
      onnx_node_name = "onehot_node"
    } : (tensor<2x2xi64>, tensor<i64>, tensor<2xf32>) -> tensor<2x10x2xf32>
    "onnx.Return"(%0) : (tensor<2x10x2xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
