// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test SimplifiedLayerNormalization E2E pipeline from real Llama-3.1-8B
// The model has one onnx.Custom(SimplifiedLayerNormalization) with constant gamma.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Custom(SimplifiedLayerNormalization) -> hip.rms_norm
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops -> LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_rms_norm
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16> {onnx.name = "/model/embed_tokens/Gather/output_0"}) -> (tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/input_layernorm/output_0"}) {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4096xf16>} : () -> tensor<4096xf16>
    %1 = "onnx.Custom"(%arg0, %0) {function_name = "SimplifiedLayerNormalization", axis = -1 : si64, domain_name = "", epsilon = 9.99999974E-6 : f32, onnx_node_name = "/model/layers.0/input_layernorm/LayerNorm", stash_type = 1 : si64} : (tensor<1x128x4096xf16>, tensor<4096xf16>) -> tensor<1x128x4096xf16>
    "onnx.Return"(%1) : (tensor<1x128x4096xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
