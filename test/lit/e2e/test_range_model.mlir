// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Verifies end-to-end lowering for ONNX Range through custom-kernel runtime call.
// 1. convert-onnx-to-hip: onnx.Range -> hip.range
// 2. convert-hip-to-llvm: hip.range -> llvm.call @wrap_range
// 3. generate-interface: inference_init/compute/cleanup/metadata

// CHECK-DAG: llvm.func @wrap_range
// CHECK-DAG: llvm.func @hipdnn_ep_state_reset_error_flag
// CHECK-DAG: llvm.func @hipdnn_ep_state_read_and_clear_error_flag
// CHECK-LABEL: llvm.func @inference_compute
// CHECK: llvm.call @hipdnn_ep_state_reset_error_flag
// CHECK: llvm.call @main_graph
// CHECK: llvm.call @hipdnn_ep_state_read_and_clear_error_flag
// CHECK-NOT: onnx.Range

module {
  func.func @main_graph(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?xi32> {
    %0 = "onnx.Range"(%arg0, %arg1, %arg2) {onnx_node_name = "/range"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
    "onnx.Return"(%0) : (tensor<?xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
