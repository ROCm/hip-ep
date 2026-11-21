// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test bf16 broadcast Div E2E full pipeline.
// A different-shape bf16 Div ([1,512,896] / [1,512,1], the per-token scale
// shape of bf16 LLMs) is rewritten on the ONNX path to Reciprocal + Mul.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Div -> onnx.Reciprocal + onnx.Mul
//    -> hip.reciprocal + hip.mul
// 2. convert-hip-to-llvm: hip.reciprocal -> wrap_power, hip.mul ->
//    wrap_elementwise; no wrap_div call is left
// 3. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// Reciprocal: wrap_power(..., data_type, alpha=0, beta=1, gamma=-1), bf16 (2).
// CHECK-NOT: llvm.call @wrap_div(
// CHECK: [[RDT:%[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : f64)
// CHECK-NEXT: llvm.mlir.constant(1.000000e+00 : f64)
// CHECK-NEXT: llvm.mlir.constant(-1.000000e+00 : f64)
// CHECK-NEXT: llvm.call @wrap_power({{.*}}, [[RDT]], {{.*}})
// Mul: wrap_elementwise(..., data_type, tensor_op), bf16 (2).
// CHECK-NOT: llvm.call @wrap_div(
// CHECK: llvm.call @hipdnn_ep_alloc_output(
// CHECK: [[MDT:%[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: [[OP:%[0-9]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_elementwise({{.*}}, [[MDT]], [[OP]])
// CHECK-NOT: llvm.call @wrap_div(
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Div
module {
  func.func @main_graph(%arg0: tensor<1x512x896xbf16> {onnx.name = "div_a"},
                        %arg1: tensor<1x512x1xbf16> {onnx.name = "div_b"})
      -> (tensor<1x512x896xbf16> {onnx.name = "div_out"}) {
    %0 = "onnx.Div"(%arg0, %arg1) {onnx_node_name = "div_broadcast"} : (tensor<1x512x896xbf16>, tensor<1x512x1xbf16>) -> tensor<1x512x896xbf16>
    "onnx.Return"(%0) : (tensor<1x512x896xbf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
