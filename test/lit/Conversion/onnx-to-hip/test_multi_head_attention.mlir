// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify com.microsoft.MultiHeadAttention (via onnx.Custom) is correctly
// lowered to hip.multi_head_attention in tensor-first mode.
//
// This test suite validates:
// - Custom ONNX op matching by function_name and domain_name
// - Runtime-supported separate rank-3 fp16 Q/K/V subset
// - Attribute preservation (num_heads, scale, mask_filter_value,
//   unidirectional)
// - Tensor-first DPS: output shape comes from query
//
// Test cases:
// 1. Basic self-attention with separate Q/K/V (single output)
// 2. Cross-attention with explicit K/V from encoder (single output)
// 3. Causal masking (unidirectional = 1)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

// =============================================================================
// Test 1: Basic self-attention with separate Q/K/V
// =============================================================================
module {
  func.func @main_graph(
      %query: tensor<1x128x4096xf16>,
      %key: tensor<1x128x4096xf16>,
      %value: tensor<1x128x4096xf16>)
      -> tensor<1x128x4096xf16> {

    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %out = "onnx.Custom"(%query, %key, %value)
        <{function_name = "MultiHeadAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 32 : si64,
         scale = 0.0883883461 : f32,
         mask_filter_value = -1.000000e+04 : f32,
         unidirectional = 0 : si64}
        : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
        -> tensor<1x128x4096xf16>

    // One tensor.empty() init for the single output
    // CHECK: tensor.empty() : tensor<1x128x4096xf16>
    // CHECK: hip.multi_head_attention(%[[CTX]])
    // CHECK-SAME: ins(
    // CHECK-SAME: num_heads = 32
    // Note: mask_filter_value=-1e4 and unidirectional=0 are DefaultValuedAttr in
    // HipOps.td, so MLIR's auto-generated printer omits them when they equal the
    // default. Their semantic correctness is covered by Test 4 (unidirectional=1
    // explicit, non-default) and by hip-to-llvm/test_multi_head_attention.mlir.
    // CHECK-NOT: hip.alloc

    return %out : tensor<1x128x4096xf16>
  }

  // ===========================================================================
  // Test 2: Cross-attention (key/value distinct from query)
  // ===========================================================================
  func.func @test_cross_attention(
      %query: tensor<1x64x4096xf16>,
      %key: tensor<1x128x4096xf16>,
      %value: tensor<1x128x4096xf16>)
      -> tensor<1x64x4096xf16> {

    // CHECK-LABEL: func.func @test_cross_attention
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %out = "onnx.Custom"(%query, %key, %value)
        <{function_name = "MultiHeadAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 32 : si64,
         scale = 0.0883883461 : f32}
        : (tensor<1x64x4096xf16>, tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
        -> tensor<1x64x4096xf16>

    // CHECK: hip.multi_head_attention(%[[CTX]])
    // CHECK-SAME: num_heads = 32

    return %out : tensor<1x64x4096xf16>
  }

  // ===========================================================================
  // Test 3: Causal self-attention (unidirectional = 1)
  // ===========================================================================
  func.func @test_causal(
      %query: tensor<1x128x4096xf16>,
      %key: tensor<1x128x4096xf16>,
      %value: tensor<1x128x4096xf16>)
      -> tensor<1x128x4096xf16> {

    // CHECK-LABEL: func.func @test_causal
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,

    %out = "onnx.Custom"(%query, %key, %value)
        <{function_name = "MultiHeadAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 32 : si64,
         scale = 0.0883883461 : f32,
         unidirectional = 1 : si64}
        : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
        -> tensor<1x128x4096xf16>

    // CHECK: hip.multi_head_attention(%[[CTX]])
    // CHECK-SAME: num_heads = 32
    // CHECK-SAME: unidirectional = 1

    return %out : tensor<1x128x4096xf16>
  }

  // ===========================================================================
  // Test 5: Dynamic shapes - batch and seq_len unknown at compile time
  // ===========================================================================
  func.func @test_dynamic_shape(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x4096xf16>,
      %value: tensor<?x?x4096xf16>)
      -> tensor<?x?x4096xf16> {

    // CHECK-LABEL: func.func @test_dynamic_shape
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context,
    // Use a restricted SSA-name regex for Q so the greedy `.*` doesn't capture
    // across multiple matching `: tensor<?x?x4096xf16>,` suffixes (3 args have
    // the same type here).
    // CHECK-SAME: %[[Q:[a-zA-Z0-9_]+]]: tensor<?x?x4096xf16>,

    %out = "onnx.Custom"(%query, %key, %value)
        <{function_name = "MultiHeadAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 32 : si64,
         scale = 0.0883883461 : f32}
        : (tensor<?x?x4096xf16>, tensor<?x?x4096xf16>, tensor<?x?x4096xf16>)
        -> tensor<?x?x4096xf16>

    // Dynamic dims must be extracted from the source query tensor via
    // tensor.dim before being fed into tensor.empty.
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[D0:.*]] = tensor.dim %[[Q]], %[[C0]] : tensor<?x?x4096xf16>
    // CHECK: %[[D1:.*]] = tensor.dim %[[Q]], %[[C1]] : tensor<?x?x4096xf16>
    // CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?x4096xf16>
    // CHECK: hip.multi_head_attention(%[[CTX]])
    // CHECK-SAME: num_heads = 32

    return %out : tensor<?x?x4096xf16>
  }
}
