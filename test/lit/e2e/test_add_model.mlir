// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Add E2E full pipeline: elementwise addition with broadcasting
// GPT-OSS-20B MoE router: MatMulNBits output + bias
// Input:  tensor<1x128x32xf16>  (router logits)
// Bias:   tensor<32xf16>        (router bias, broadcast)
// Output: tensor<1x128x32xf16>
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Add → hip.add
// 2. bufferize / canonicalize
// 3. convert-hip-to-llvm: hip.add → wrap_elementwise(tensor_op=1)
// 4. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_elementwise
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Add
module {
  func.func @main_graph(%arg0: tensor<1x128x32xf16> {onnx.name = "input_0"}) -> (tensor<1x128x32xf16> {onnx.name = "output_0"}) {
    %bias = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<32xf16>} : () -> tensor<32xf16>
    %0 = "onnx.Add"(%arg0, %bias) {onnx_node_name = "Add_0"} : (tensor<1x128x32xf16>, tensor<32xf16>) -> tensor<1x128x32xf16>
    "onnx.Return"(%0) : (tensor<1x128x32xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
