// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Greater E2E full pipeline.
// onnx.Greater is decomposed into hip.less(b, a), so the final LLVM IR
// contains wrap_less and the original onnx.Greater must be gone.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_less
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Greater
module {
  func.func @main_graph(%arg0: tensor<3x4xf32> {onnx.name = "a"}, %arg1: tensor<3x4xf32> {onnx.name = "b"}) -> (tensor<3x4xi1> {onnx.name = "output"}) {
    %0 = "onnx.Greater"(%arg0, %arg1) {onnx_node_name = "greater_node"} : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1>
    "onnx.Return"(%0) : (tensor<3x4xi1>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
