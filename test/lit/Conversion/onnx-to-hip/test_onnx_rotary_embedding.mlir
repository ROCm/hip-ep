// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the native ai.onnx RotaryEmbedding op (opset >= 23, domain "") lowers
// to hip.rope. This is distinct from com.microsoft.RotaryEmbedding (onnx.Custom,
// covered by test_rope.mlir). It validates the two native-op specifics:
//   1. Operand order (X, cos_cache, sin_cache, position_ids?) is remapped to
//      hip.rope's (input, position_ids, cos_cache, sin_cache).
//   2. position_ids is optional. When absent, cos/sin are precomputed 3D
//      [batch, seq, rotary_dim/2] and hip.rope is emitted WITHOUT a
//      position_ids operand.
//   3. num_heads from attribute (3D) or shape[1] (4D); rotary_embedding_dim=0
//      inferred from cos_cache last dim * 2.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===== No position_ids: precomputed 3D cos/sin (Gemma-style) =====
  func.func @gemma_rope_no_posids(%x: tensor<?x?x4096xf16>,
                                   %cos: tensor<?x?x128xf16>,
                                   %sin: tensor<?x?x128xf16>)
      -> tensor<?x?x4096xf16> {
    // num_heads=16 from attribute; rotary_dim inferred = 128 * 2 = 256.
    %y = "onnx.RotaryEmbedding"(%x, %cos, %sin) {
      interleaved = 0 : si64,
      num_heads = 16 : si64,
      rotary_embedding_dim = 0 : si64
    } : (tensor<?x?x4096xf16>, tensor<?x?x128xf16>, tensor<?x?x128xf16>)
        -> tensor<?x?x4096xf16>
    return %y : tensor<?x?x4096xf16>
  }

  // CHECK-LABEL: func.func @gemma_rope_no_posids
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x?x4096xf16>, %[[COS:.*]]: tensor<?x?x128xf16>, %[[SIN:.*]]: tensor<?x?x128xf16>) -> tensor<?x?x4096xf16>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: tensor.dim %[[X]], %[[C0]] : tensor<?x?x4096xf16>
  // CHECK: tensor.dim %[[X]], %[[C1]] : tensor<?x?x4096xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x4096xf16>
  // CHECK: hip.rope(%[[CTX]]) ins(%[[X]], %[[COS]], %[[SIN]] : tensor<?x?x4096xf16>, tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%[[INIT]] : tensor<?x?x4096xf16>) {interleaved = 0 : i64, num_heads = 16 : i64, rotary_embedding_dim = 256 : i64} : tensor<?x?x4096xf16>
  // CHECK-NOT: onnx.RotaryEmbedding

  // ===== With position_ids: 2D lookup table, operand remap =====
  func.func @rope_with_posids(%x: tensor<1x128x4096xf16>,
                               %cos: tensor<131072x64xf16>,
                               %sin: tensor<131072x64xf16>,
                               %pos: tensor<1x128xi64>)
      -> tensor<1x128x4096xf16> {
    // num_heads=32 from attribute; rotary_dim inferred = 64 * 2 = 128.
    %y = "onnx.RotaryEmbedding"(%x, %cos, %sin, %pos) {
      interleaved = 0 : si64,
      num_heads = 32 : si64,
      rotary_embedding_dim = 0 : si64
    } : (tensor<1x128x4096xf16>, tensor<131072x64xf16>,
         tensor<131072x64xf16>, tensor<1x128xi64>)
        -> tensor<1x128x4096xf16>
    return %y : tensor<1x128x4096xf16>
  }

  // CHECK-LABEL: func.func @rope_with_posids
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x128x4096xf16>, %[[COS:.*]]: tensor<131072x64xf16>, %[[SIN:.*]]: tensor<131072x64xf16>, %[[POS:.*]]: tensor<1x128xi64>) -> tensor<1x128x4096xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x128x4096xf16>
  // Operand remap: (X, cos, sin, pos) -> (input, pos, cos, sin).
  // CHECK: hip.rope(%[[CTX]]) ins(%[[X]], %[[POS]], %[[COS]], %[[SIN]] : tensor<1x128x4096xf16>, tensor<1x128xi64>, tensor<131072x64xf16>, tensor<131072x64xf16>) outs(%[[INIT]] : tensor<1x128x4096xf16>) {interleaved = 0 : i64, num_heads = 32 : i64, rotary_embedding_dim = 128 : i64} : tensor<1x128x4096xf16>
  // CHECK-NOT: onnx.RotaryEmbedding

  // ===== 4D BNSH input, no position_ids: num_heads from shape[1] =====
  func.func @rope_4d_no_posids(%x: tensor<1x16x8x256xf16>,
                                %cos: tensor<1x8x128xf16>,
                                %sin: tensor<1x8x128xf16>)
      -> tensor<1x16x8x256xf16> {
    // num_heads=0 -> inferred from shape[1]=16; rotary_dim = 128 * 2 = 256.
    %y = "onnx.RotaryEmbedding"(%x, %cos, %sin) {
      interleaved = 0 : si64,
      num_heads = 0 : si64,
      rotary_embedding_dim = 0 : si64
    } : (tensor<1x16x8x256xf16>, tensor<1x8x128xf16>, tensor<1x8x128xf16>)
        -> tensor<1x16x8x256xf16>
    return %y : tensor<1x16x8x256xf16>
  }

  // CHECK-LABEL: func.func @rope_4d_no_posids
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x16x8x256xf16>, %[[COS:.*]]: tensor<1x8x128xf16>, %[[SIN:.*]]: tensor<1x8x128xf16>) -> tensor<1x16x8x256xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x16x8x256xf16>
  // CHECK: hip.rope(%[[CTX]]) ins(%[[X]], %[[COS]], %[[SIN]] : tensor<1x16x8x256xf16>, tensor<1x8x128xf16>, tensor<1x8x128xf16>) outs(%[[INIT]] : tensor<1x16x8x256xf16>) {interleaved = 0 : i64, num_heads = 16 : i64, rotary_embedding_dim = 256 : i64} : tensor<1x16x8x256xf16>
  // CHECK-NOT: onnx.RotaryEmbedding
}
