// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Slice E2E full pipeline (decompose path).
// onnx.Slice with constant starts/ends and positive unit stride is
// rewritten to tensor.extract_slice, which bufferizes to memref.subview —
// no runtime call. Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Slice -> tensor.extract_slice
// 2. bufferize:           tensor.extract_slice -> memref.subview
// 3. convert-hip-to-llvm: full lowering to LLVM IR
// 4. generate-interface:  inference_init / compute / cleanup / metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Slice
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<4x6xf32> {onnx.name = "input"})
      -> (tensor<2x6xf32> {onnx.name = "output"}) {
    %starts = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends   = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes   = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
    %steps  = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps)
        {onnx_node_name = "slice_node"}
        : (tensor<4x6xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<2x6xf32>
    "onnx.Return"(%0) : (tensor<2x6xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
