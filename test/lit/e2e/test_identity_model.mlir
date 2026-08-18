// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Identity E2E full pipeline.
// onnx.Identity is a pass-through: the conversion forwards its input SSA
// value to every user (cheaper than even a full-range memref.subview view),
// so the op disappears in convert-onnx-to-hip. The output-allocator pass then
// creates one exact public destination and copies the forwarded input into it.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Identity -> (eliminated, value forwarded)
// 2. hip-use-output-allocator: allocate output slot 0 and materialize the copy.
// 3. generate-interface: Create inference_init/compute/cleanup/metadata.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Identity
// CHECK-LABEL: llvm.func private @main_graph_internal
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: llvm.call @wrap_hipMemcpyAsync
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
