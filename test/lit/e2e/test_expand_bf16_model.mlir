// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test bf16 Expand E2E full pipeline.
// Covers the bf16 broadcast used by bf16 LLMs (e.g. Qwen
// torch.aten.broadcast_to on the grouped KV heads, rank 5).
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Expand -> hip.expand
// 2. convert-hip-to-llvm: hip.expand -> wrap_expand with bf16
// 3. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// in_rank and out_rank (5) are followed by data_type
// HIPDNN_EP_DATATYPE_BFLOAT16 (2), the last argument.
// CHECK: llvm.func @wrap_expand
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: llvm.mlir.constant(5 : i64) : i64
// CHECK-NEXT: llvm.mlir.constant(5 : i64) : i64
// CHECK-NEXT: [[DT:%[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_expand({{.*}}, [[DT]])
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Expand
module {
  func.func @main_graph(%arg0: tensor<1x2x1x512x64xbf16> {onnx.name = "kv_in"},
                        %arg1: tensor<5xi64> {onnx.name = "shape"})
      -> (tensor<1x2x7x512x64xbf16> {onnx.name = "kv_out"}) {
    %0 = "onnx.Expand"(%arg0, %arg1) {onnx_node_name = "expand_kv"} : (tensor<1x2x1x512x64xbf16>, tensor<5xi64>) -> tensor<1x2x7x512x64xbf16>
    "onnx.Return"(%0) : (tensor<1x2x7x512x64xbf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
