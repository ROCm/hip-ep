// RUN: hip-mlir-opt %s --torch-hipdnn-pipeline | FileCheck %s

// Test Torch Sigmoid E2E full pipeline: sigmoid activation
// Input:  tensor<2x64xf32>
// Output: tensor<2x64xf32>
//
// Verifies the complete torch-hipdnn-pipeline:
// 1. convert-torch-to-hip: torch.aten.sigmoid → hip.sigmoid
// 2. bufferize / canonicalize
// 3. convert-hip-to-llvm: hip.sigmoid → wrap_miopenActivationForward
// 4. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenActivationForward
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: torch.aten.sigmoid
module {
  func.func @main_graph(%arg0: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = "torch.aten.sigmoid"(%arg0) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}
