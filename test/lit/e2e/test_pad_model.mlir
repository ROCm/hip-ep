// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_pad
// pads are routed to host via hip.transfer: the pipeline lowers that to a stack
// host buffer + an async D2H + a plain stream sync (NOT the internal wrap_pad
// D2H that the old runtime did).
// CHECK-DAG: llvm.func @wrap_hipMemcpyD2H
// CHECK-DAG: llvm.func @wrap_hipStreamSynchronize
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Pad
module {
  func.func @main_graph(%arg0: tensor<3x4xf32> {onnx.name = "data"}, %arg1: tensor<4xi64> {onnx.name = "pads"}) -> (tensor<5x6xf32> {onnx.name = "output"}) {
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Pad"(%arg0, %arg1, %none, %none) {mode = "constant", onnx_node_name = "pad_node"} : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<5x6xf32>
    "onnx.Return"(%0) : (tensor<5x6xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
