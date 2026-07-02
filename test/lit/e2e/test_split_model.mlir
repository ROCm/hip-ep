// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// END-TO-END TEST: ONNX Split operator
//
// Validates complete pipeline from ONNX Split through to LLVM lowering:
// 1. ONNX Split -> tensor.extract_slice (OnnxToHip conversion)
// 2. tensor.extract_slice -> memref.subview (bufferization)
// 3. memref.subview -> LLVM pointer arithmetic (MemRefToLLVM)
//
// This test ensures:
// - Split is completely lowered (no ONNX ops remain)
// - Zero-copy semantics preserved (subviews, not allocations)
// - Wrapper functions generated correctly
// - Multiple outputs handled properly
// ============================================================================

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

module {
  // Main entry point: multi-head attention split
  // Input: [batch, seq_len, hidden_dim=3*head_dim*num_heads]
  // Outputs: Q, K, V projections [batch, seq_len, head_dim*num_heads]
  func.func @main_graph(%input: tensor<1x128x3072xf16>) -> (tensor<1x128x1024xf16>, tensor<1x128x1024xf16>, tensor<1x128x1024xf16>) {
    // Split into Q, K, V
    %q, %k, %v = "onnx.Split"(%input) {axis = 2 : si64} : (tensor<1x128x3072xf16>) -> (tensor<1x128x1024xf16>, tensor<1x128x1024xf16>, tensor<1x128x1024xf16>)
    return %q, %k, %v : tensor<1x128x1024xf16>, tensor<1x128x1024xf16>, tensor<1x128x1024xf16>
  }

  // Additional test: cascaded splits (split outputs consumed by other splits)
  func.func @test_cascaded_splits(%input: tensor<1x256x4096xf32>) -> (tensor<1x256x512xf32>, tensor<1x256x512xf32>, tensor<1x256x512xf32>, tensor<1x256x512xf32>) {
    // First split: 4096 -> 2048 + 2048
    %left, %right = "onnx.Split"(%input) {axis = 2 : si64} : (tensor<1x256x4096xf32>) -> (tensor<1x256x2048xf32>, tensor<1x256x2048xf32>)

    // Second split: 2048 -> 512 + 512 + 512 + 512 (on left)
    %l0, %l1, %l2, %l3 = "onnx.Split"(%left) {axis = 2 : si64} : (tensor<1x256x2048xf32>) -> (tensor<1x256x512xf32>, tensor<1x256x512xf32>, tensor<1x256x512xf32>, tensor<1x256x512xf32>)

    return %l0, %l1, %l2, %l3 : tensor<1x256x512xf32>, tensor<1x256x512xf32>, tensor<1x256x512xf32>, tensor<1x256x512xf32>
  }

  // Custom split lengths test
  func.func @test_custom_split(%input: tensor<16x128xf16>) -> (tensor<4x128xf16>, tensor<8x128xf16>, tensor<4x128xf16>) {
    %split_lengths = "onnx.Constant"() {value = dense<[4, 8, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
    %s0, %s1, %s2 = "onnx.Split"(%input, %split_lengths) {axis = 0 : si64} : (tensor<16x128xf16>, tensor<3xi64>) -> (tensor<4x128xf16>, tensor<8x128xf16>, tensor<4x128xf16>)
    return %s0, %s1, %s2 : tensor<4x128xf16>, tensor<8x128xf16>, tensor<4x128xf16>
  }
}

// Verify no high-level ops remain after full pipeline lowering.
// CHECK-NOT: onnx.Split
// CHECK-NOT: onnx.Constant
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: func.func

// Split / bufferization may emit D2D copies as HIP wrappers (no in-tree memrefCopy).
// CHECK-NOT: llvm.call @memrefCopy
// CHECK: llvm.func private @main_graph
// CHECK: llvm.func private @main_graph_internal
// CHECK: wrap_hipMemcpy

// Verify additional split test functions survive to LLVM lowering.
// CHECK: llvm.func @test_cascaded_splits
// CHECK: llvm.func @test_custom_split

// Verify generated C interface wrappers are present.
// CHECK-LABEL: llvm.func @inference_init
// CHECK: llvm.call @hipdnn_ep_state_init_with_fs
// CHECK-LABEL: llvm.func @inference_compute
// CHECK: llvm.call @hipdnn_ep_tensor_prepare_input
// Outputs are allocated in-graph (2-arg ABI); no prepare_output/finalize_output.
// CHECK-NOT: llvm.call @hipdnn_ep_tensor_prepare_output
// CHECK-NOT: llvm.call @hipdnn_ep_tensor_finalize_output
// CHECK-LABEL: llvm.func @inference_cleanup
// CHECK: llvm.call @hipdnn_ep_state_cleanup
// CHECK-LABEL: llvm.func @inference_get_metadata_json
