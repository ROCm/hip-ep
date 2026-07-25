// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Identity E2E full pipeline.
// onnx.Identity is a pass-through: the conversion forwards its input SSA
// value to every user (cheaper than even a full-range memref.subview view),
// so the op disappears entirely in convert-onnx-to-hip.  No HIP dialect op,
// no HipToLLVM lowering, and no runtime symbol are emitted.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Identity -> (eliminated, value forwarded)
// 2. generate-interface: Create inference_init/compute/cleanup/metadata.
//
// The downstream buffer-results-to-out-params + memory-pooling passes are
// responsible for materialising any input->output buffer copy that the
// runtime needs when an SSA-forwarded input also has to leave the graph
// as a named output; this conversion does not need to reason about it.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Identity
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<2x3x4xf32> {onnx.name = "input"})
      -> (tensor<2x3x4xf32> {onnx.name = "output"}) {
    %0 = "onnx.Identity"(%arg0) {onnx_node_name = "identity_node"}
        : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    "onnx.Return"(%0) : (tensor<2x3x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
