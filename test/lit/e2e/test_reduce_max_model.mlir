// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test ReduceMax E2E full pipeline
// The model has one onnx.ReduceMax with i64 tensors (keepdims=1, scalar axis)
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.ReduceMax → hip.reduce_max
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_reduce_max
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.ReduceMax
module {
  func.func @main_graph(%arg0: tensor<1x128xi64> {onnx.name = "data"}, %arg1: tensor<i64> {onnx.name = "axes"}) -> (tensor<1x1xi64> {onnx.name = "output"}) {
    %0 = "onnx.ReduceMax"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64, onnx_node_name = "reduce_max_node"} : (tensor<1x128xi64>, tensor<i64>) -> tensor<1x1xi64>
    "onnx.Return"(%0) : (tensor<1x1xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
