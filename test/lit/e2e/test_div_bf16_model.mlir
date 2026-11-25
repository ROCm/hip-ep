// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test bf16 Div E2E full pipeline.
// A same-shape bf16 Div lowers to wrap_div (a broadcast one becomes
// Reciprocal + Mul on the ONNX path). bf16 LLMs divide in bf16, e.g. the
// logits / attention scaling of Granite.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Div -> hip.div
// 2. convert-hip-to-llvm: hip.div -> wrap_div with bf16
// 3. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// data_type is HIPDNN_EP_DATATYPE_BFLOAT16 (2), the last wrap_div argument.
// CHECK: llvm.func @wrap_div
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: [[DT:%[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_div({{.*}}, [[DT]])
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Div
module {
  func.func @main_graph(%arg0: tensor<1x512x896xbf16> {onnx.name = "div_a"},
                        %arg1: tensor<1x512x896xbf16> {onnx.name = "div_b"})
      -> (tensor<1x512x896xbf16> {onnx.name = "div_out"}) {
    %0 = "onnx.Div"(%arg0, %arg1) {onnx_node_name = "div_same_shape"} : (tensor<1x512x896xbf16>, tensor<1x512x896xbf16>) -> tensor<1x512x896xbf16>
    "onnx.Return"(%0) : (tensor<1x512x896xbf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
