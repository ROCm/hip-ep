// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Sub E2E full pipeline from real Llama-3.1-8B attention mask subgraph
// This IR was imported from Sub_fix.onnx via onnx-mlir
// The model has one onnx.Sub with i64 tensors (includes scalar input)
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Sub → hip.elementwise_sub
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_elementwise_sub
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Sub
module {
  func.func @main_graph(%arg0: tensor<1x1xi64> {onnx.name = "/model/attn_mask_reformat/attn_mask_subgraph/ReduceSum/output_0"}, %arg1: tensor<i64> {onnx.name = "/model/constants/TensorProto.INT64/1D/1"}) -> (tensor<1x1xi64> {onnx.name = "/model/attn_mask_reformat/attn_mask_subgraph/Sub/output_0"}) {
    %0 = "onnx.Sub"(%arg0, %arg1) {onnx_node_name = "/model/attn_mask_reformat/attn_mask_subgraph/Sub"} : (tensor<1x1xi64>, tensor<i64>) -> tensor<1x1xi64>
    "onnx.Return"(%0) : (tensor<1x1xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
