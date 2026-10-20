// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --hip-group-matmul-nbits only groups members whose result is
// [A's leading axes, N] in A's element type (the group sizes its fresh
// destinations from A) and that run on the same context.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-group-matmul-nbits --split-input-file | FileCheck %s

// The rank-2 member cannot take a destination shaped from A; the other two
// still group.
// CHECK-LABEL: func.func @flattened_member
// CHECK: hip.matmul_nbits_group
// CHECK-SAME: Ns = array<i64: 32, 16>
// CHECK: hip.matmul_nbits(%{{.*}}) ins(%{{.*}}) outs(%{{.*}} : tensor<?x16xf16>)
// CHECK-NOT: hip.matmul_nbits_group
func.func @flattened_member(%ctx: !hip.context, %x: tensor<1x?x64xf16>,
    %w0: tensor<32x2x16xui8>, %s0: tensor<32x2xf16>,
    %w1: tensor<16x2x16xui8>, %s1: tensor<16x2xf16>, %t2: index)
    -> (tensor<1x?x32xf16>, tensor<1x?x16xf16>, tensor<?x16xf16>) {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %x, %c1 : tensor<1x?x64xf16>
  %e0 = tensor.empty(%t) : tensor<1x?x32xf16>
  %y0 = hip.matmul_nbits(%ctx) ins(%x, %w0, %s0 : tensor<1x?x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e0 : tensor<1x?x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x?x32xf16>
  %e1 = tensor.empty(%t) : tensor<1x?x16xf16>
  %y1 = hip.matmul_nbits(%ctx) ins(%x, %w1, %s1 : tensor<1x?x64xf16>, tensor<16x2x16xui8>, tensor<16x2xf16>)
      outs(%e1 : tensor<1x?x16xf16>)
      {K = 64 : i64, N = 16 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x?x16xf16>
  %e2 = tensor.empty(%t2) : tensor<?x16xf16>
  %y2 = hip.matmul_nbits(%ctx) ins(%x, %w1, %s1 : tensor<1x?x64xf16>, tensor<16x2x16xui8>, tensor<16x2xf16>)
      outs(%e2 : tensor<?x16xf16>)
      {K = 64 : i64, N = 16 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<?x16xf16>
  return %y0, %y1, %y2 : tensor<1x?x32xf16>, tensor<1x?x16xf16>, tensor<?x16xf16>
}

// -----

// Projections on different contexts never share a group.
// CHECK-LABEL: func.func @two_contexts
// CHECK-NOT: hip.matmul_nbits_group
// CHECK-COUNT-2: hip.matmul_nbits(
func.func @two_contexts(%ctx0: !hip.context, %ctx1: !hip.context,
    %x: tensor<1x1x64xf16>, %w0: tensor<32x2x16xui8>, %s0: tensor<32x2xf16>)
    -> (tensor<1x1x32xf16>, tensor<1x1x32xf16>) {
  %e0 = tensor.empty() : tensor<1x1x32xf16>
  %y0 = hip.matmul_nbits(%ctx0) ins(%x, %w0, %s0 : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e0 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %y1 = hip.matmul_nbits(%ctx1) ins(%x, %w0, %s0 : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e1 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  return %y0, %y1 : tensor<1x1x32xf16>, tensor<1x1x32xf16>
}
