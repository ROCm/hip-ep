// RUN: hip-mlir-opt %s --torch-hipdnn-pipeline | FileCheck %s

// Test Torch Scaled Dot-Product Attention E2E via GQA backend
// Q: tensor<1x4x128xf16>  (batch x seq x num_heads*head_dim) BSD format
// K: tensor<1x4x128xf16>
// V: tensor<1x4x128xf16>
// Output: tensor<1x4x128xf16>
//
// Note: Uses BSD input format directly (matching GQA's native layout)
// rather than BHSD, to avoid rank-4 transpose limitation in hip.transpose.
// In a real model pipeline, the reshape from BHSD to BSD happens upstream.
//
// Verifies:
// 1. torch.aten.scaled_dot_product_attention -> hip.gqa
// 2. Full pipeline lowering to LLVM

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_group_query_attention
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK-NOT: torch.aten.scaled_dot_product_attention
module {
  func.func @main_graph(%arg0: tensor<1x4x128xf16>, %arg1: tensor<1x4x128xf16>, %arg2: tensor<1x4x128xf16>) -> tensor<1x4x128xf16> {
    %0 = "torch.aten.scaled_dot_product_attention"(%arg0, %arg1, %arg2) : (tensor<1x4x128xf16>, tensor<1x4x128xf16>, tensor<1x4x128xf16>) -> tensor<1x4x128xf16>
    return %0 : tensor<1x4x128xf16>
  }
}
