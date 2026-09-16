// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Erf E2E full pipeline
// Erf: Y = erf(X), lowered via wrap_erf

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_erf
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Erf
module {
  func.func @main_graph(%arg0: tensor<1x128x200x200xf32> {onnx.name = "input"}) -> (tensor<1x128x200x200xf32> {onnx.name = "output"}) {
    %0 = "onnx.Erf"(%arg0) {onnx_node_name = "erf_node"} : (tensor<1x128x200x200xf32>) -> tensor<1x128x200x200xf32>
    "onnx.Return"(%0) : (tensor<1x128x200x200xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
