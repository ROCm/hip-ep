// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Concat E2E full pipeline (decompose path).
// onnx.Concat is rewritten to tensor.empty + N x tensor.insert_slice,
// which bufferize to memref.subview + memref.copy against a pooled
// output buffer — no Concat-specific runtime kernel. Verifies the
// complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Concat -> tensor.empty + tensor.insert_slice
// 2. bufferize:           tensor.insert_slice -> memref.subview + memref.copy
// 3. convert-hip-to-llvm: full lowering to LLVM IR
// 4. generate-interface:  inference_init / compute / cleanup / metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Concat
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<2x4xf32> {onnx.name = "a"},
                        %arg1: tensor<3x4xf32> {onnx.name = "b"})
      -> (tensor<5x4xf32> {onnx.name = "output"}) {
    %0 = "onnx.Concat"(%arg0, %arg1) {axis = 0 : si64, onnx_node_name = "concat_node"}
        : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<5x4xf32>
    "onnx.Return"(%0) : (tensor<5x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
