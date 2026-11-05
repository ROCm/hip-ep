// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test bf16 Neg E2E full pipeline.
// Covers the bf16 Neg used by bf16 LLMs (e.g. Qwen torch.aten.neg in the
// rotary embedding, rank 4).
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Neg -> hip.neg
// 2. convert-hip-to-llvm: hip.neg -> wrap_neg with bf16
// 3. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// data_type is HIPDNN_EP_DATATYPE_BFLOAT16 (2), the last argument.
// CHECK: llvm.func @wrap_neg
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: [[DT:%[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_neg({{.*}}, [[DT]])
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Neg
module {
  func.func @main_graph(%arg0: tensor<1x14x512x64xbf16> {onnx.name = "neg_in"})
      -> (tensor<1x14x512x64xbf16> {onnx.name = "neg_out"}) {
    %0 = "onnx.Neg"(%arg0) {onnx_node_name = "neg"} : (tensor<1x14x512x64xbf16>) -> tensor<1x14x512x64xbf16>
    "onnx.Return"(%0) : (tensor<1x14x512x64xbf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
