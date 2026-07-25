// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Size E2E full pipeline -- DYNAMIC input shape.
//
// Companion to test_size_model.mlir (static path). For dynamic input
// shapes, onnx.Size is NOT folded; it becomes a hip.size DPS op which
// lowers to a runtime wrap_size call. This test verifies the full
// pipeline (convert-onnx-to-hip + bufferize + memory-pooling +
// convert-hip-to-llvm + generate-interface) handles the dynamic path
// end-to-end and produces a model.dll-shaped LLVM module that:
//   * declares wrap_size,
//   * keeps the inference interface intact (init/compute/cleanup/metadata),
//   * does NOT leave any leftover onnx.Size.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_size
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Size
module {
  func.func @main_graph(%arg0: tensor<?x?xf32> {onnx.name = "input"})
      -> (tensor<i64> {onnx.name = "size"}) {
    %0 = "onnx.Size"(%arg0) {onnx_node_name = "size_node"}
        : (tensor<?x?xf32>) -> tensor<i64>
    "onnx.Return"(%0) : (tensor<i64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
