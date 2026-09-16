// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Tanh E2E full pipeline -- DYNAMIC input shape.
//
// Companion to test_tanh_model.mlir (static path). With fully dynamic leading
// dims the EP still compiles the dynamic graph: onnx.Tanh lowers to a hip.tanh
// DPS op whose num_elements is computed at runtime, then to a
// wrap_tanh call. This verifies the full pipeline
// (convert-onnx-to-hip + bufferize + memory-pooling + convert-hip-to-llvm +
// generate-interface) handles the dynamic path end-to-end and leaves no
// leftover onnx.Tanh.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_tanh
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Tanh
module {
  func.func @main_graph(%arg0: tensor<?x?x512xf16> {onnx.name = "input"}) -> (tensor<?x?x512xf16> {onnx.name = "output"}) {
    %0 = "onnx.Tanh"(%arg0) {onnx_node_name = "tanh_node"} : (tensor<?x?x512xf16>) -> tensor<?x?x512xf16>
    "onnx.Return"(%0) : (tensor<?x?x512xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
