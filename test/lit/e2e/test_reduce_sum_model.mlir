// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test ReduceSum E2E full pipeline from real Llama-3.1-8B attention mask subgraph
// This IR was imported from ReduceSum_fix.onnx via onnx-mlir
// The model has one onnx.ReduceSum with i64 tensors (keepdims=1, scalar axis)
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.ReduceSum → hip.reduce_sum
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_reduce_sum
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.ReduceSum
module {
  func.func @main_graph(%arg0: tensor<1x128xi64> {onnx.name = "attention_mask"}, %arg1: tensor<i64> {onnx.name = "/model/constants/TensorProto.INT64/1D/1"}) -> (tensor<1x1xi64> {onnx.name = "/model/attn_mask_reformat/attn_mask_subgraph/ReduceSum/output_0"}) {
    %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64, onnx_node_name = "/model/attn_mask_reformat/attn_mask_subgraph/ReduceSum"} : (tensor<1x128xi64>, tensor<i64>) -> tensor<1x1xi64>
    "onnx.Return"(%0) : (tensor<1x1xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
