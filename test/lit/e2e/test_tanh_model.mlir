// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Tanh E2E full pipeline.
// The model has one onnx.Tanh with f16 tensors, mirroring the standalone
// Tanh used by Gemma-2 logit soft-capping (logits = cap * tanh(logits/cap)).
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Tanh → hip.tanh
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenActivationForward
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Tanh
module {
  func.func @main_graph(%arg0: tensor<1x128x14336xf16> {onnx.name = "/lm_head/softcap/Div/output_0"}) -> (tensor<1x128x14336xf16> {onnx.name = "/lm_head/softcap/Tanh/output_0"}) {
    %0 = "onnx.Tanh"(%arg0) {onnx_node_name = "/lm_head/softcap/Tanh"} : (tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
    "onnx.Return"(%0) : (tensor<1x128x14336xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
