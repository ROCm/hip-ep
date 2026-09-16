// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_floor
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Floor
module {
  func.func @main_graph(%arg0: tensor<3x4xf32> {onnx.name = "input"})
      -> (tensor<3x4xf32> {onnx.name = "output"}) {
    %0 = "onnx.Floor"(%arg0) {onnx_node_name = "floor_node"}
        : (tensor<3x4xf32>) -> tensor<3x4xf32>
    "onnx.Return"(%0) : (tensor<3x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
