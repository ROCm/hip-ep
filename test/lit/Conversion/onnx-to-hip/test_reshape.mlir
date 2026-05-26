// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Reshape is correctly lowered to tensor.expand_shape /
// tensor.collapse_shape (zero-cost standard MLIR metadata ops).
//
// Reshape is a zero-cost operation that reinterprets shape/strides without
// moving data.  No custom HIP dialect op or kernel is needed.
//
// This test validates:
// - Static expand (split dim):     3D -> 4D
// - Static collapse (merge dims):  4D -> 3D
// - Rank-reducing collapse:        3D -> 2D (merge leading dims)
// - Same-rank reshape:             3D -> 3D (collapse to 1D then expand)
// - Identity reshape:              same shape, replaced by identity
//
// Model: GPT-OSS-20B head split/merge reshapes
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16>) -> tensor<1x128x4096xf16> {
    return %arg0 : tensor<1x128x4096xf16>
  }

  // --- Static expand: split last dim ---
  func.func @test_reshape_expand(%data: tensor<1x128x4096xf16>) -> tensor<1x128x32x128xf16> {
    %shape = "onnx.Constant"() {value = dense<[1, 128, 32, 128]> : tensor<4xi64>} : () -> tensor<4xi64>
    %result = "onnx.Reshape"(%data, %shape) : (tensor<1x128x4096xf16>, tensor<4xi64>) -> tensor<1x128x32x128xf16>
    return %result : tensor<1x128x32x128xf16>
  }

  // --- Static collapse: merge last two dims ---
  func.func @test_reshape_collapse(%data: tensor<1x128x32x128xf16>) -> tensor<1x128x4096xf16> {
    %shape = "onnx.Constant"() {value = dense<[1, 128, 4096]> : tensor<3xi64>} : () -> tensor<3xi64>
    %result = "onnx.Reshape"(%data, %shape) : (tensor<1x128x32x128xf16>, tensor<3xi64>) -> tensor<1x128x4096xf16>
    return %result : tensor<1x128x4096xf16>
  }

  // --- Rank-reducing collapse: merge leading dims (from Reshape_seq128.onnx) ---
  func.func @test_reshape_rank_reduce(%data: tensor<1x128x32xf16>) -> tensor<128x32xf16> {
    %shape = "onnx.Constant"() {value = dense<[-1, 32]> : tensor<2xi64>} : () -> tensor<2xi64>
    %result = "onnx.Reshape"(%data, %shape) {allowzero = 0 : si64} : (tensor<1x128x32xf16>, tensor<2xi64>) -> tensor<128x32xf16>
    return %result : tensor<128x32xf16>
  }

  // --- Same-rank reshape: different shape, same total elements ---
  func.func @test_reshape_same_rank(%data: tensor<2x3x4096xf16>) -> tensor<6x1x4096xf16> {
    %shape = "onnx.Constant"() {value = dense<[6, 1, 4096]> : tensor<3xi64>} : () -> tensor<3xi64>
    %result = "onnx.Reshape"(%data, %shape) : (tensor<2x3x4096xf16>, tensor<3xi64>) -> tensor<6x1x4096xf16>
    return %result : tensor<6x1x4096xf16>
  }

  // --- Identity reshape ---
  func.func @test_reshape_identity(%data: tensor<1x128x4096xf16>) -> tensor<1x128x4096xf16> {
    %shape = "onnx.Constant"() {value = dense<[1, 128, 4096]> : tensor<3xi64>} : () -> tensor<3xi64>
    %result = "onnx.Reshape"(%data, %shape) : (tensor<1x128x4096xf16>, tensor<3xi64>) -> tensor<1x128x4096xf16>
    return %result : tensor<1x128x4096xf16>
  }

  // ==========================================================================
  // Dynamic-shape Reshape (Qwen3.5 text.onnx support)
  // ==========================================================================

  // (1) Rank-1[?] -> rank-2[?, ?] with shape vector from from_elements.
  // Mirrors the Qwen text.onnx pattern:
  //   range_out: tensor<?xi64>
  //   shape    : tensor.from_elements %B, %S : tensor<2xi64>
  //   pos      : Reshape(range_out, shape) -> tensor<?x?xi64>
  func.func @test_reshape_rank1_to_rank2_dyn(
      %data: tensor<?xi64>, %B: i64, %S: i64) -> tensor<?x?xi64> {
    %shape = tensor.from_elements %B, %S : tensor<2xi64>
    %r = "onnx.Reshape"(%data, %shape)
        : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x?xi64>
    return %r : tensor<?x?xi64>
  }

  // (2) Same-rank dyn reshape with DIFFERENT static-dim layout:
  // rank-3 [?, ?, 256] -> rank-3 [?, ?, 4096]. This is the canonical
  // text.onnx pattern: [B, S*H_q, 256] -> [B, S, H_q*256] where H_q*256
  // = 4096. The input and output types differ (256 vs 4096 on the
  // trailing dim) so the no-op short-circuit cannot fire; the converter
  // must take the collapse-to-1D + expand path and read the two leading
  // dyn dims from the shape operand.
  func.func @test_reshape_same_rank_dyn(
      %data: tensor<?x?x256xf16>, %B: i64, %S: i64) -> tensor<?x?x4096xf16> {
    %c4096 = arith.constant 4096 : i64
    %shape = tensor.from_elements %B, %S, %c4096 : tensor<3xi64>
    %r = "onnx.Reshape"(%data, %shape)
        : (tensor<?x?x256xf16>, tensor<3xi64>) -> tensor<?x?x4096xf16>
    return %r : tensor<?x?x4096xf16>
  }

  // (3) Rank-1[?] -> rank-2[?, 256] single-dyn-per-group: should use
  // the fast path via tensor.dim (NOT tensor.extract from the shape
  // operand). The dyn dim is recoverable as input.dim(0) / 256.
  func.func @test_reshape_rank1_to_rank2_static_inner(
      %data: tensor<?xi64>) -> tensor<?x256xi64> {
    %shape = "onnx.Constant"() {value = dense<[-1, 256]> : tensor<2xi64>}
        : () -> tensor<2xi64>
    %r = "onnx.Reshape"(%data, %shape)
        : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x256xi64>
    return %r : tensor<?x256xi64>
  }
}

// CHECK-LABEL: func.func @test_reshape_expand
// CHECK-NOT: onnx.Reshape
// CHECK: tensor.expand_shape
// CHECK-SAME: {{\[\[}}0], [1], [2, 3]]

// CHECK-LABEL: func.func @test_reshape_collapse
// CHECK-NOT: onnx.Reshape
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[\[}}0], [1], [2, 3]]

// CHECK-LABEL: func.func @test_reshape_rank_reduce
// CHECK-NOT: onnx.Reshape
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[\[}}0, 1], [2]]

// CHECK-LABEL: func.func @test_reshape_same_rank
// CHECK-NOT: onnx.Reshape
// CHECK: tensor.collapse_shape
// CHECK: tensor.expand_shape

// CHECK-LABEL: func.func @test_reshape_identity
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK: return

// ============================================================================
// Dynamic-shape Reshape checks (added for Qwen3.5 text.onnx support).
// ============================================================================

// (1) Multi-dyn expand: must emit tensor.extract on the shape operand
// (one per output dim) and feed the index-cast results to
// tensor.expand_shape's output_shape operands.
// CHECK-LABEL: func.func @test_reshape_rank1_to_rank2_dyn
// CHECK-NOT: onnx.Reshape
// CHECK: tensor.from_elements
// CHECK: tensor.extract
// CHECK: arith.index_cast
// CHECK: tensor.extract
// CHECK: arith.index_cast
// CHECK: tensor.expand_shape

// (2) Same-rank dyn: collapse-to-1D then expand using extracts from
// the shape operand. Two extract -> index_cast pairs are emitted for
// the two dyn dims; the static dim is a compile-time attr in the
// expand_shape output_shape.
// CHECK-LABEL: func.func @test_reshape_same_rank_dyn
// CHECK-NOT: onnx.Reshape
// CHECK: tensor.collapse_shape
// CHECK: tensor.extract
// CHECK: arith.index_cast
// CHECK: tensor.extract
// CHECK: arith.index_cast
// CHECK: tensor.expand_shape

// (3) Fast path: NO tensor.extract on the shape operand; the dyn dim is
// computed as tensor.dim(input, 0) / 256.
// CHECK-LABEL: func.func @test_reshape_rank1_to_rank2_static_inner
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: tensor.extract
// CHECK: tensor.dim
// CHECK: tensor.expand_shape
