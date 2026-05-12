// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Neg E2E full pipeline
// Neg: y = -x, lowered via unified unary elementwise path (wrap_neg)

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_neg
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Neg
module {
  func.func @main_graph(%arg0: tensor<3x4xf32> {onnx.name = "input"}) -> (tensor<3x4xf32> {onnx.name = "output"}) {
    %0 = "onnx.Neg"(%arg0) {onnx_node_name = "neg_node"} : (tensor<3x4xf32>) -> tensor<3x4xf32>
    "onnx.Return"(%0) : (tensor<3x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
