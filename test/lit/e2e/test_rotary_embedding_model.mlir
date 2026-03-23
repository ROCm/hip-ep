// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test RotaryEmbedding E2E pipeline from real Llama-3.1-8B attention subgraph
// The model has one onnx.Custom(RotaryEmbedding) with constant cos/sin tables.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Custom(RotaryEmbedding) → hip.rope
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_rotary_embedding
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/attn/q_proj/MatMul/output_0"}, %arg1: tensor<1x128xi64> {onnx.name = "position_ids"}) -> (tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/attn/q_rotary/RotaryEmbedding/output_0"}) {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<131072x64xf16>} : () -> tensor<131072x64xf16>
    %1 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<131072x64xf16>} : () -> tensor<131072x64xf16>
    %2 = "onnx.Custom"(%arg0, %arg1, %0, %1) {function_name = "RotaryEmbedding", domain_name = "com.microsoft", interleaved = 0 : si64, num_heads = 0 : si64, onnx_node_name = "/model/layers.0/attn/q_rotary/RotaryEmbedding", rotary_embedding_dim = 0 : si64} : (tensor<1x128x4096xf16>, tensor<1x128xi64>, tensor<131072x64xf16>, tensor<131072x64xf16>) -> tensor<1x128x4096xf16>
    "onnx.Return"(%2) : (tensor<1x128x4096xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
