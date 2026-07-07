// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_compress
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Compress
module {
  func.func @main_graph(%arg0: tensor<3x2xf32> {onnx.name = "input"},
                          %arg1: tensor<3xi1> {onnx.name = "condition"})
      -> (tensor<2x2xf32> {onnx.name = "output"}) {
    %0 = "onnx.Compress"(%arg0, %arg1) {
      axis = 0 : si64,
      onnx_node_name = "compress_node"
    } : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<2x2xf32>
    "onnx.Return"(%0) : (tensor<2x2xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
