// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Binary einsum through the full pipeline:
//   onnx.Einsum -> transpose / reshape + hip.matmul -> wrap_hipblasLtMatmul
//
// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_hipblasLtMatmul
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Einsum
module {
  func.func @main_graph(%arg0: tensor<2x3x4x5xf16> {onnx.name = "A"}, %arg1: tensor<3x6x5xf16> {onnx.name = "B"}) -> (tensor<2x3x4x6xf16> {onnx.name = "Y"}) {
    %0 = "onnx.Einsum"(%arg0, %arg1) {equation = "bhwc,hkc->bhwk", onnx_node_name = "einsum_node"} : (tensor<2x3x4x5xf16>, tensor<3x6x5xf16>) -> tensor<2x3x4x6xf16>
    "onnx.Return"(%0) : (tensor<2x3x4x6xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
