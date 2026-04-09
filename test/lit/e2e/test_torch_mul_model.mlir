// RUN: hip-mlir-opt %s --torch-hipdnn-pipeline | FileCheck %s

// Test Torch Mul E2E full pipeline: elementwise multiplication
// Input A: tensor<2x3x4xf32>
// Input B: tensor<2x3x4xf32>
// Output:  tensor<2x3x4xf32>
//
// Verifies the complete torch-hipdnn-pipeline:
// 1. convert-torch-to-hip: torch.aten.mul.Tensor → hip.mul
// 2. bufferize / canonicalize
// 3. convert-hip-to-llvm: hip.mul → wrap_miopenOpTensor(tensor_op=0)
// 4. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenOpTensor
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: torch.aten.mul
module {
  func.func @main_graph(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = "torch.aten.mul.Tensor"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}
