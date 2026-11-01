// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test IsNaN E2E full pipeline.
// IsNaN: Y = isnan(X), f16 input, ui8 boolean output, lowered via wrap_isnan.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_isnan
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.IsNaN
module {
  func.func @main_graph(%arg0: tensor<24x20x1x1xf16> {onnx.name = "input"}) -> (tensor<24x20x1x1xui8> {onnx.name = "output"}) {
    %0 = "onnx.IsNaN"(%arg0) {onnx_node_name = "isnan_node"} : (tensor<24x20x1x1xf16>) -> tensor<24x20x1x1xui8>
    "onnx.Return"(%0) : (tensor<24x20x1x1xui8>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
