// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Cast E2E full pipeline from real Llama-3.1-8B attention mask subgraph
// This IR was imported from Cast_fix.onnx via onnx-mlir
// The model has one onnx.Cast converting i64 scalar to i32
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Cast → hip.cast
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_cast
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Cast
module {
  func.func @main_graph(%arg0: tensor<i64> {onnx.name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather/output_0"}) -> (tensor<i32> {onnx.name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast/output_0"}) {
    %0 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = i32, onnx_node_name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast"} : (tensor<i64>) -> tensor<i32>
    "onnx.Return"(%0) : (tensor<i32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
