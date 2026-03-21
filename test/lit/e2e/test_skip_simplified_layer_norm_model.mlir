// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test SkipSimplifiedLayerNormalization E2E pipeline from real Llama-3.1-8B
// The model has one onnx.Custom(SkipSimplifiedLayerNormalization) with constant gamma.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Custom(SkipSimplifiedLayerNormalization) → hip.skip_rms_norm
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 2
// CHECK: llvm.func @wrap_skip_simplified_layer_norm
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16> {onnx.name = "/model/embed_tokens/Gather/output_0"}, %arg1: tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/attn/o_proj/MatMul/output_0"}) -> (tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/post_attention_layernorm/output_0"}, tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/post_attention_layernorm/output_3"}) {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4096xf16>} : () -> tensor<4096xf16>
    %1:4 = "onnx.Custom"(%arg0, %arg1, %0) {function_name = "SkipSimplifiedLayerNormalization", domain_name = "com.microsoft", epsilon = 9.99999974E-6 : f32, onnx_node_name = "/model/layers.0/post_attention_layernorm/SkipLayerNorm"} : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<4096xf16>) -> (tensor<1x128x4096xf16>, none, none, tensor<1x128x4096xf16>)
    "onnx.Return"(%1#0, %1#3) : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
