// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Sin E2E full pipeline
// Sin: Y = sin(X), lowered via wrap_sin

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_sin
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Sin
module {
  func.func @main_graph(%arg0: tensor<3x4xf32> {onnx.name = "input"}) -> (tensor<3x4xf32> {onnx.name = "output"}) {
    %0 = "onnx.Sin"(%arg0) {onnx_node_name = "sin_node"} : (tensor<3x4xf32>) -> tensor<3x4xf32>
    "onnx.Return"(%0) : (tensor<3x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
