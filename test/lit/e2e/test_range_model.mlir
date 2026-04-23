// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Verifies end-to-end lowering for ONNX Range through custom-kernel runtime call.
// 1. convert-onnx-to-hip: onnx.Range -> hip.range
// 2. convert-hip-to-llvm: hip.range -> llvm.call @wrap_range
// 3. generate-interface: inference_init/compute/cleanup/metadata

// CHECK: llvm.func @wrap_range
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Range

module {
  func.func @main_graph(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?xi32> {
    %0 = "onnx.Range"(%arg0, %arg1, %arg2) {onnx_node_name = "/range"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
    "onnx.Return"(%0) : (tensor<?xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
