// RUN: hip-mlir-opt %s --torch-hipdnn-pipeline | FileCheck %s

// Test Torch Add E2E full pipeline: elementwise addition
// Input:  tensor<1x128x32xf16>
// Bias:   tensor<1x128x32xf16> (same shape, no broadcast)
// Output: tensor<1x128x32xf16>
//
// Verifies the complete torch-hipdnn-pipeline:
// 1. convert-torch-to-hip: torch.aten.add.Tensor → hip.add
// 2. bufferize / canonicalize
// 3. convert-hip-to-llvm: hip.add → wrap_miopenOpTensor(tensor_op=1)
// 4. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenOpTensor
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: torch.aten.add
module {
  func.func @main_graph(%arg0: tensor<1x128x32xf16>, %arg1: tensor<1x128x32xf16>) -> tensor<1x128x32xf16> {
    %int1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
    %0 = "torch.aten.add.Tensor"(%arg0, %arg1, %int1) : (tensor<1x128x32xf16>, tensor<1x128x32xf16>, !torch.int) -> tensor<1x128x32xf16>
    return %0 : tensor<1x128x32xf16>
  }
}
