// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Flatten E2E full pipeline.
// onnx.Flatten decomposes to tensor.collapse_shape (and tensor.expand_shape
// for axis = 0 / axis = r corner cases). Both bufferize to zero-cost
// memref descriptor edits — no HIP kernel and no runtime symbol are emitted.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Flatten -> tensor.collapse_shape
// 2. one-shot-bufferize: tensor -> memref descriptor ops
// 3. generate-interface: Create inference_init/compute/cleanup/metadata.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Flatten
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<2x3x4x5xf32> {onnx.name = "input"})
      -> (tensor<2x60xf32> {onnx.name = "output"}) {
    %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64, onnx_node_name = "flatten_node"}
        : (tensor<2x3x4x5xf32>) -> tensor<2x60xf32>
    "onnx.Return"(%0) : (tensor<2x60xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
