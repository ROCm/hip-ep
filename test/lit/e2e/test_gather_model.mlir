// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Gather E2E full pipeline from real Llama-3.1-8B attention mask subgraph
// This IR was imported from Gather_fix.onnx via onnx-mlir
// The model has one onnx.Gather with i64 tensors (axis=0, scalar index)
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Gather → hip.gather
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_gather
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Gather
module {
  func.func @main_graph(%arg0: tensor<2xi64> {onnx.name = "/model/attn_mask_reformat/attn_mask_subgraph/Shape/output_0"}, %arg1: tensor<i64> {onnx.name = "/model/constants/TensorProto.INT64/0D/1"}) -> (tensor<i64> {onnx.name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather/output_0"}) {
    %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64, onnx_node_name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather"} : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    "onnx.Return"(%0) : (tensor<i64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
