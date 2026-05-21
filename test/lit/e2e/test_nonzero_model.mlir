// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test NonZero E2E full pipeline
// This IR represents an ONNX NonZero operation imported via onnx-mlir
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.NonZero -> hip.nonzero (with slot_id attr)
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops -> LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata
// 6. compose-dim-specs: NonZero's data-dependent dim becomes a
//    RuntimeSlot; the module-level dyn slots counter is non-zero; and
//    the dyn-slot ABI entry points (inference_dyn_slot_get_*, _reset)
//    are emitted so that the EP host-side resolver can pull the
//    published dim and buffer post-compute.

// CHECK: module attributes {
// CHECK-DAG: hipdnn.input_count = 1
// CHECK-DAG: hipdnn.output_count = 1
// CHECK-DAG: hipdnn.dyn_dim_slots_count = 1
// CHECK-DAG: llvm.func @wrap_nonzero
// CHECK-DAG: llvm.func @inference_init
// CHECK-DAG: llvm.func @inference_compute
// CHECK-DAG: llvm.func @inference_cleanup
// CHECK-DAG: llvm.func @inference_get_metadata_json
// CHECK-DAG: llvm.func @inference_dyn_slot_get_dim
// CHECK-DAG: llvm.func @inference_dyn_slot_get_buffer
// CHECK-DAG: llvm.func @inference_dyn_slot_reset
// CHECK-NOT: onnx.NonZero
module {
  func.func @main_graph(%arg0: tensor<3x4xi1> {onnx.name = "input"}) -> (tensor<2x?xi64> {onnx.name = "output"}) {
    %0 = "onnx.NonZero"(%arg0) {onnx_node_name = "nonzero_node"} : (tensor<3x4xi1>) -> tensor<2x?xi64>
    "onnx.Return"(%0) : (tensor<2x?xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
