// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Cos E2E full pipeline
// Cos: Y = cos(X), lowered via wrap_cos

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_cos
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Cos
module {
  func.func @main_graph(%arg0: tensor<3x4xf32> {onnx.name = "input"}) -> (tensor<3x4xf32> {onnx.name = "output"}) {
    %0 = "onnx.Cos"(%arg0) {onnx_node_name = "cos_node"} : (tensor<3x4xf32>) -> tensor<3x4xf32>
    "onnx.Return"(%0) : (tensor<3x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
