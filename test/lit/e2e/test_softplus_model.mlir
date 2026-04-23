// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Softplus E2E full pipeline
// The model has one onnx.Softplus with f16 tensors
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Softplus → hip.softplus
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata
//
// Softplus directly uses MIOpen's wrap_miopenActivationForward()
// with HIPDNN_EP_ACTIVATION_SOFTPLUS (activation_mode=3).

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenActivationForward
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Softplus
module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16> {onnx.name = "input"}) -> (tensor<1x128x4096xf16> {onnx.name = "output"}) {
    %0 = "onnx.Softplus"(%arg0) {onnx_node_name = "softplus_node"} : (tensor<1x128x4096xf16>) -> tensor<1x128x4096xf16>
    "onnx.Return"(%0) : (tensor<1x128x4096xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
