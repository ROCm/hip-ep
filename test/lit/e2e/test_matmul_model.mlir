// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test MatMul E2E pipeline from real Llama-3.1-8B k_proj subgraph
// The model has one onnx.MatMul with f16 tensors and a constant weight matrix.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.MatMul → hip.matmul
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_hipblasLtMatmul(
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.MatMul
module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/input_layernorm/output_0"}) -> (tensor<1x128x1024xf16> {onnx.name = "/model/layers.0/attn/k_proj/MatMul/output_0"}) {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4096x1024xf16>} : () -> tensor<4096x1024xf16>
    %1 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "/model/layers.0/attn/k_proj/MatMul"} : (tensor<1x128x4096xf16>, tensor<4096x1024xf16>) -> tensor<1x128x1024xf16>
    "onnx.Return"(%1) : (tensor<1x128x1024xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
