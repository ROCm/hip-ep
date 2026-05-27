// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test NonZero E2E full pipeline
// This IR represents an ONNX NonZero operation imported via onnx-mlir
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.NonZero -> hip.nonzero
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops -> LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_nonzero
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.NonZero
module {
  func.func @main_graph(%arg0: tensor<3x4xi1> {onnx.name = "input"}) -> (tensor<2x?xi64> {onnx.name = "output"}) {
    %0 = "onnx.NonZero"(%arg0) {onnx_node_name = "nonzero_node"} : (tensor<3x4xi1>) -> tensor<2x?xi64>
    "onnx.Return"(%0) : (tensor<2x?xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
