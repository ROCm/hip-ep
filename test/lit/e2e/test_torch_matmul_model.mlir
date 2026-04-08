// RUN: hip-mlir-opt %s --torch-hipdnn-pipeline | FileCheck %s

// Test Torch MatMul E2E full pipeline: matrix multiplication
// Input A: tensor<4x8xf32>
// Input B: tensor<8x16xf32>
// Output:  tensor<4x16xf32>
//
// Verifies the complete torch-hipdnn-pipeline:
// 1. convert-torch-to-hip: torch.aten.mm → hip.matmul
// 2. bufferize / canonicalize
// 3. convert-hip-to-llvm: hip.matmul → wrap_hipblasLtMatmul
// 4. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_hipblasLtMatmul
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: torch.aten.mm
module {
  func.func @main_graph(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %0 = "torch.aten.mm"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }
}
