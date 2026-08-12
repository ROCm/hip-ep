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

  // --- Rank-0 scalar → rank-1 1-element (Qwen vision shape-arith chain) ---
  // onnx.Reshape on a tensor<i64> scalar to tensor<1xi64> can't be expressed
  // via tensor.expand_shape (source rank 0 can't supply a reassoc group).
  // Lower via tensor.extract + tensor.from_elements.
  func.func @test_reshape_scalar_to_1d(%data: tensor<i64>) -> tensor<1xi64> {
    %shape = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.Reshape"(%data, %shape) {allowzero = 0 : si64} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    return %result : tensor<1xi64>
  }

  // --- Rank-0 scalar → rank-3 1x1x1 (defensive: degenerate higher-rank case) ---
  func.func @test_reshape_scalar_to_3d_unit(%data: tensor<f16>) -> tensor<1x1x1xf16> {
    %shape = "onnx.Constant"() {value = dense<[1, 1, 1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %result = "onnx.Reshape"(%data, %shape) {allowzero = 0 : si64} : (tensor<f16>, tensor<3xi64>) -> tensor<1x1x1xf16>
    return %result : tensor<1x1x1xf16>
  }

  // --- Multi-dynamic expand: [bs*ss, H] -> [bs, ss, H] ---
  // The source dim only holds the PRODUCT bs*ss, so the split has to come from
  // the shape operand.  Deriving both extents positionally from the source
  // yields [bs*ss, bs*ss, H], which is how every Gemma-4 decoder layer's input
  // norm ended up dispatched over ss^2 rows.
  func.func @test_reshape_expand_multi_dyn(%data: tensor<?x2816xf16>,
                                           %ref: tensor<?x?x2816xf16>)
      -> tensor<?x?x2816xf16> {
    %shape = "onnx.Shape"(%ref) : (tensor<?x?x2816xf16>) -> tensor<3xi64>
    %result = "onnx.Reshape"(%data, %shape) {allowzero = 0 : si64} : (tensor<?x2816xf16>, tensor<3xi64>) -> tensor<?x?x2816xf16>
    return %result : tensor<?x?x2816xf16>
  }

  // --- Multi-dynamic expand with an opaque shape operand ---
  // Nothing here reveals the split, so the conversion must decline to build an
  // expand_shape and hand the runtime shape vector to tensor.reshape instead.
  func.func @test_reshape_expand_multi_dyn_opaque(%data: tensor<?x2816xf16>,
                                                  %shape: tensor<3xi64>)
      -> tensor<?x?x2816xf16> {
    %result = "onnx.Reshape"(%data, %shape) {allowzero = 0 : si64} : (tensor<?x2816xf16>, tensor<3xi64>) -> tensor<?x?x2816xf16>
    return %result : tensor<?x?x2816xf16>
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

// A runtime (non-constant) scalar must be read to the host with a synchronized
// hip.readback_scalar, NOT a bare tensor.extract: the latter lowers to an
// unsynchronized host load of a device buffer and reads stale memory on
// true-device-memory targets (see ReadbackScalar.h).
// CHECK-LABEL: func.func @test_reshape_scalar_to_1d
// CHECK: hip.readback_scalar
// CHECK: tensor.from_elements
// CHECK-NOT: onnx.Reshape

// CHECK-LABEL: func.func @test_reshape_scalar_to_3d_unit
// CHECK: hip.readback_scalar
// CHECK: tensor.from_elements
// CHECK: tensor.expand_shape
// CHECK-NOT: onnx.Reshape

// The two dynamic output extents must be DISTINCT dims of the rank-3 reference,
// never the same source dim twice.
// CHECK-LABEL: func.func @test_reshape_expand_multi_dyn
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[BS:.*]] = tensor.dim %[[REF:.*]], %[[C0]] : tensor<?x?x2816xf16>
// CHECK: %[[SS:.*]] = tensor.dim %[[REF]], %[[C1]] : tensor<?x?x2816xf16>
// CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [%[[BS]], %[[SS]], 2816]
// CHECK-NOT: onnx.Reshape

// CHECK-LABEL: func.func @test_reshape_expand_multi_dyn_opaque
// CHECK-NOT: tensor.expand_shape
// CHECK: tensor.reshape
// CHECK-NOT: onnx.Reshape
