// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX RotaryEmbedding (com.microsoft domain) is correctly lowered
// to hip.rope operation in tensor-first mode.
//
// This test validates:
// - Custom operation lowering (onnx.Custom -> hip.rope)
// - Domain check: only "com.microsoft" domain is converted
// - Rotary embedding attributes: interleaved, num_heads, rotary_embedding_dim
// - Attribute inference (num_heads=0, rotary_dim=0 inferred from shapes)
// - f16 element type support
// - 3D tensor (1x128x4096) rotation with position indices
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
//
// Model: Llama-3.1-8B rotary position embedding
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x128x4096xf16>,
                         %position_ids: tensor<1x128xi64>,
                         %cos_cache: tensor<131072x64xf16>,
                         %sin_cache: tensor<131072x64xf16>)
      -> tensor<1x128x4096xf16> {
    // Test attribute inference: 0 means "infer from tensor shapes"
    // Expected: rotary_dim = cos_cache.shape[-1] * 2 = 64 * 2 = 128
    //           num_heads = input.shape[-1] / rotary_dim = 4096 / 128 = 32
    %output = "onnx.Custom"(%input, %position_ids, %cos_cache, %sin_cache) {
      function_name = "RotaryEmbedding",
      domain_name = "com.microsoft",
      interleaved = 0 : si64,
      num_heads = 0 : si64,
      rotary_embedding_dim = 0 : si64
    } : (tensor<1x128x4096xf16>, tensor<1x128xi64>,
         tensor<131072x64xf16>, tensor<131072x64xf16>)
        -> tensor<1x128x4096xf16>
    return %output : tensor<1x128x4096xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x4096xf16>, %[[POS_IDS:.*]]: tensor<1x128xi64>, %[[COS:.*]]: tensor<131072x64xf16>, %[[SIN:.*]]: tensor<131072x64xf16>) -> tensor<1x128x4096xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x128x4096xf16>
  // CHECK: hip.rope(%[[CTX]]) ins(%[[INPUT]], %[[POS_IDS]], %[[COS]], %[[SIN]] : tensor<1x128x4096xf16>, tensor<1x128xi64>, tensor<131072x64xf16>, tensor<131072x64xf16>) outs(%[[INIT]] : tensor<1x128x4096xf16>) {interleaved = 0 : i64, num_heads = 32 : i64, rotary_embedding_dim = 128 : i64} : tensor<1x128x4096xf16>
  // CHECK-NOT: onnx.Custom
  // CHECK-NOT: hip.alloc
  // CHECK-NOT: hip.copy

  // ===== Dynamic shape test =====

  func.func @dynamic_rope(%input: tensor<?x?x?xf16>,
                          %position_ids: tensor<?x?xi64>,
                          %cos_cache: tensor<?x?xf16>,
                          %sin_cache: tensor<?x?xf16>)
      -> tensor<?x?x?xf16> {
    %output = "onnx.Custom"(%input, %position_ids, %cos_cache, %sin_cache) {
      function_name = "RotaryEmbedding",
      domain_name = "com.microsoft",
      interleaved = 0 : si64,
      num_heads = 32 : si64,
      rotary_embedding_dim = 128 : si64
    } : (tensor<?x?x?xf16>, tensor<?x?xi64>,
         tensor<?x?xf16>, tensor<?x?xf16>)
        -> tensor<?x?x?xf16>
    return %output : tensor<?x?x?xf16>
  }

  // CHECK-LABEL: func.func @dynamic_rope
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<?x?x?xf16>, %[[POS_IDS:.*]]: tensor<?x?xi64>, %[[COS:.*]]: tensor<?x?xf16>, %[[SIN:.*]]: tensor<?x?xf16>) -> tensor<?x?x?xf16>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %{{.*}} = tensor.dim %[[INPUT]], %[[C0]] : tensor<?x?x?xf16>
  // CHECK: %{{.*}} = tensor.dim %[[INPUT]], %[[C1]] : tensor<?x?x?xf16>
  // CHECK: %{{.*}} = tensor.dim %[[INPUT]], %[[C2]] : tensor<?x?x?xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty({{.*}}, {{.*}}, {{.*}}) : tensor<?x?x?xf16>
  // CHECK: hip.rope(%[[CTX]]) ins(%[[INPUT]], %[[POS_IDS]], %[[COS]], %[[SIN]] : tensor<?x?x?xf16>, tensor<?x?xi64>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%[[INIT]] : tensor<?x?x?xf16>) {interleaved = 0 : i64, num_heads = 32 : i64, rotary_embedding_dim = 128 : i64} : tensor<?x?x?xf16>
  // CHECK-NOT: onnx.Custom
  // CHECK-NOT: hip.alloc
  // CHECK-NOT: hip.copy

  // ===== Explicit attributes (no inference) =====

  func.func @explicit_attrs_rope(%input: tensor<1x128x4096xf16>,
                                  %position_ids: tensor<1x128xi64>,
                                  %cos_cache: tensor<131072x64xf16>,
                                  %sin_cache: tensor<131072x64xf16>)
      -> tensor<1x128x4096xf16> {
    %output = "onnx.Custom"(%input, %position_ids, %cos_cache, %sin_cache) {
      function_name = "RotaryEmbedding",
      domain_name = "com.microsoft",
      interleaved = 1 : si64,
      num_heads = 32 : si64,
      rotary_embedding_dim = 128 : si64
    } : (tensor<1x128x4096xf16>, tensor<1x128xi64>,
         tensor<131072x64xf16>, tensor<131072x64xf16>)
        -> tensor<1x128x4096xf16>
    return %output : tensor<1x128x4096xf16>
  }

  // CHECK-LABEL: func.func @explicit_attrs_rope
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x4096xf16>, %[[POS_IDS:.*]]: tensor<1x128xi64>, %[[COS:.*]]: tensor<131072x64xf16>, %[[SIN:.*]]: tensor<131072x64xf16>) -> tensor<1x128x4096xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x128x4096xf16>
  // CHECK: hip.rope(%[[CTX]]) ins(%[[INPUT]], %[[POS_IDS]], %[[COS]], %[[SIN]] : tensor<1x128x4096xf16>, tensor<1x128xi64>, tensor<131072x64xf16>, tensor<131072x64xf16>) outs(%[[INIT]] : tensor<1x128x4096xf16>) {interleaved = 1 : i64, num_heads = 32 : i64, rotary_embedding_dim = 128 : i64} : tensor<1x128x4096xf16>
  // CHECK-NOT: onnx.Custom
  // CHECK-NOT: hip.alloc
  // CHECK-NOT: hip.copy
}
