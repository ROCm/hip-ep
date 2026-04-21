// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test CausalConvWithState E2E full pipeline
// This IR represents a com.microsoft.CausalConvWithState operation imported
// via onnx-mlir as an onnx.Custom op. Used by Gated DeltaNet (Qwen3.5) and
// Mamba (Jamba, FalconMamba) models.
//
// The model has one onnx.Custom with f16 tensors:
//   inputs:  input (B=1, C=64, L=128), weight (64, 1, k=4),
//            bias (64), past_state (1, 64, k-1=3)
//   outputs: output (1, 64, 128), present_state (1, 64, 3)
//   attrs:   activation = "silu", ndim = 1
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Custom (CausalConvWithState)
//                         → hip.causal_conv_with_state
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output and present_state buffers into single alloc
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 4
// CHECK-SAME: hipdnn.output_count = 2
// CHECK: llvm.func @wrap_causal_conv_with_state
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(
      %arg0: tensor<1x64x128xf16> {onnx.name = "input"},
      %arg1: tensor<64x1x4xf16> {onnx.name = "weight"},
      %arg2: tensor<64xf16> {onnx.name = "bias"},
      %arg3: tensor<1x64x3xf16> {onnx.name = "past_state"})
      -> (tensor<1x64x128xf16> {onnx.name = "output"},
          tensor<1x64x3xf16> {onnx.name = "present_state"}) {
    %0:2 = "onnx.Custom"(%arg0, %arg1, %arg2, %arg3) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      onnx_node_name = "causal_conv_node",
      activation = "silu",
      ndim = 1 : si64
    } : (tensor<1x64x128xf16>, tensor<64x1x4xf16>, tensor<64xf16>, tensor<1x64x3xf16>)
      -> (tensor<1x64x128xf16>, tensor<1x64x3xf16>)
    "onnx.Return"(%0#0, %0#1) : (tensor<1x64x128xf16>, tensor<1x64x3xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
