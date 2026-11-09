// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_isinf
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.IsInf
module {
  func.func @main_graph(%arg0: tensor<3x4xf16> {onnx.name = "input"})
      -> (tensor<3x4xi1> {onnx.name = "output"}) {
    %0 = "onnx.IsInf"(%arg0) {onnx_node_name = "isinf_node"}
        : (tensor<3x4xf16>) -> tensor<3x4xi1>
    "onnx.Return"(%0) : (tensor<3x4xi1>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
