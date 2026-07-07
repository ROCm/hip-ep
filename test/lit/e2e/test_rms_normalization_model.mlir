// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test RMSNormalization E2E pipeline.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.RMSNormalization -> hip.rms_norm
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops -> LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenT5LayerNormForward
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.RMSNormalization
module {
  func.func @main_graph(
      %arg0: tensor<1x128x4096xf16> {onnx.name = "input"},
      %arg1: tensor<4096xf16> {onnx.name = "scale"})
      -> (tensor<1x128x4096xf16> {onnx.name = "output"}) {
    %0 = "onnx.RMSNormalization"(%arg0, %arg1) {
      axis = -1 : si64,
      epsilon = 9.99999974E-6 : f32,
      onnx_node_name = "rms_norm_node",
      stash_type = 1 : si64
    } : (tensor<1x128x4096xf16>, tensor<4096xf16>)
        -> tensor<1x128x4096xf16>
    "onnx.Return"(%0) : (tensor<1x128x4096xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
