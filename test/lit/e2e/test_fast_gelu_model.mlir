// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test FastGelu E2E full pipeline
// This IR represents a com.microsoft.FastGelu operation imported via onnx-mlir
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Custom(FastGelu) -> hip.fast_gelu
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops -> LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_fast_gelu
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Custom
// CHECK-NOT: onnx.NoValue
module {
  func.func @main_graph(
      %arg0: tensor<1x128x768xf16> {onnx.name = "data"},
      %arg1: tensor<768xf16> {onnx.name = "bias"})
      -> (tensor<1x128x768xf16> {onnx.name = "output"}) {
    %0 = "onnx.Custom"(%arg0, %arg1) {
      function_name = "FastGelu",
      domain_name = "com.microsoft",
      onnx_node_name = "fast_gelu_node"
    } : (tensor<1x128x768xf16>, tensor<768xf16>) -> tensor<1x128x768xf16>
    "onnx.Return"(%0) : (tensor<1x128x768xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
