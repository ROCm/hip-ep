// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Pre-lowering rewrites that decompose ONNX ops into compositions the
// existing converters already handle. Tests pin two invariants today:
//
//  1. AveragePoolToReshapeMean emits an `onnx.Transpose` with
//     `perm = [0, 1, 2, 4, 3, 5]` between the 6-D Reshape and the
//     ReduceMean. The transpose moves the K-K patch axes to the trailing
//     position because `hip_reduce_sum` only reduces a contiguous
//     suffix; without this transpose the subsequent reduce would
//     silently sum the wrong contiguous block and produce garbage
//     downstream (NaN cascade visible at model output).
//
//  2. BroadcastDivToMulReciprocal turns `onnx.Div(a, b)` with
//     differing operand shapes into `Mul(a, Reciprocal(b))`, because
//     `hip.div` is a flat element-wise kernel with no broadcast
//     support. Reciprocal preserves rhs shape; the subsequent Mul
//     broadcasts via the existing path (no kernel-level OOB read).
//
//  3. PointwiseConvToMatMul turns a LARGE-Cin static 1x1 `onnx.Conv`
//     (stride 1, no pad, no dilation, group 1) into Reshape + `onnx.MatMul`
//     (+ broadcast bias Add), routing it through hipBLASLt — so the result
//     lowers to `hip.matmul`, NOT `hip.conv`. A SMALL-Cin 1x1 conv is instead
//     let through to `hip.conv` (its HIP->LLVM lowering routes it to the fused
//     GEMM+bias custom kernel; that routing is checked in the hip-to-llvm
//     lit suite). A dynamic-shape 1x1 conv also stays a `hip.conv` (the
//     emitted MatMul's batch dim cannot be sized correctly).

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }

  // AveragePool with kernel == stride (canonical patch-pooling shape).
  // Trigger the AveragePoolToReshapeMean rewrite.
  func.func @test_avgpool_to_reshape_mean(
      %x: tensor<?x1152x16x16xf16>) -> tensor<?x1152x4x4xf16> {
    %y = "onnx.AveragePool"(%x) {
      kernel_shape = [4 : si64, 4 : si64],
      strides = [4 : si64, 4 : si64],
      auto_pad = "NOTSET",
      pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64]
    } : (tensor<?x1152x16x16xf16>) -> tensor<?x1152x4x4xf16>
    return %y : tensor<?x1152x4x4xf16>
  }

  // Broadcast Div with rhs lower-rank — trigger BroadcastDivToMulReciprocal.
  // Shape pattern matches the canonical Gemma-3 projector RMSNorm divide.
  func.func @test_broadcast_div_to_mul_reciprocal(
      %x: tensor<?x256x1152xf16>, %y: tensor<?x256x1xf16>)
      -> tensor<?x256x1152xf16> {
    %z = "onnx.Div"(%x, %y)
        : (tensor<?x256x1152xf16>, tensor<?x256x1xf16>)
        -> tensor<?x256x1152xf16>
    return %z : tensor<?x256x1152xf16>
  }

  // LARGE-Cin static 1x1 conv with bias — trigger PointwiseConvToMatMul. Cin
  // (128) is above the fused-kernel threshold, so it must lower to a hipBLASLt
  // matmul (+ broadcast bias add), never a MIOpen conv.
  func.func @test_pointwise_conv_large_cin_to_matmul(
      %x: tensor<1x128x64x64xf16>, %w: tensor<256x128x1x1xf16>,
      %b: tensor<256xf16>) -> tensor<1x256x64x64xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [1 : si64, 1 : si64],
      strides = [1 : si64, 1 : si64],
      pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64],
      dilations = [1 : si64, 1 : si64],
      group = 1 : si64
    } : (tensor<1x128x64x64xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>)
      -> tensor<1x256x64x64xf16>
    return %y : tensor<1x256x64x64xf16>
  }

  // SMALL-Cin static 1x1 conv with bias — Cin (16) is at/below the fused-kernel
  // threshold, so PointwiseConvToMatMul lets it through to ConvToHip. It stays
  // a hip.conv here (the wrap_pointwise_conv routing happens in HIP->LLVM), and
  // must NOT be rewritten into a matmul.
  func.func @test_pointwise_conv_small_cin_to_conv(
      %x: tensor<1x16x64x64xf16>, %w: tensor<256x16x1x1xf16>,
      %b: tensor<256xf16>) -> tensor<1x256x64x64xf16> {
    %y = "onnx.Conv"(%x, %w, %b) {
      kernel_shape = [1 : si64, 1 : si64],
      strides = [1 : si64, 1 : si64],
      pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64],
      dilations = [1 : si64, 1 : si64],
      group = 1 : si64
    } : (tensor<1x16x64x64xf16>, tensor<256x16x1x1xf16>, tensor<256xf16>)
      -> tensor<1x256x64x64xf16>
    return %y : tensor<1x256x64x64xf16>
  }

  // Dynamic-batch 1x1 conv — PointwiseConvToMatMul must NOT fire (the emitted
  // MatMul cannot size a dynamic batch dim); it falls through to the MIOpen
  // hip.conv path.
  func.func @test_pointwise_conv_dynamic_stays_conv(
      %x: tensor<?x16x64x64xf16>, %w: tensor<256x16x1x1xf16>)
      -> tensor<?x256x64x64xf16> {
    %y = "onnx.Conv"(%x, %w) {
      kernel_shape = [1 : si64, 1 : si64],
      strides = [1 : si64, 1 : si64],
      pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64],
      dilations = [1 : si64, 1 : si64],
      group = 1 : si64
    } : (tensor<?x16x64x64xf16>, tensor<256x16x1x1xf16>)
      -> tensor<?x256x64x64xf16>
    return %y : tensor<?x256x64x64xf16>
  }
}

// AvgPool decomp emits an onnx.Transpose with perm=[0,1,2,4,3,5] BEFORE
// it lowers (the Transpose converter then turns it into the hip-dialect
// equivalent). FileCheck on the perm attribute is the most reliable
// signature — the surrounding Reshape / ReduceSum / Div forms are
// implementation details that may move around.
// CHECK-LABEL: func.func @test_avgpool_to_reshape_mean
// CHECK-DAG: perm = [0, 1, 2, 4, 3, 5]
// CHECK-NOT: onnx.AveragePool

// Broadcast Div: onnx.Div is gone, replaced by Reciprocal + Mul which
// then lower to hip.reciprocal + hip.mul. Either form (onnx.* mid-
// pipeline or hip.* end-pipeline) proves the rewrite fired — check for
// hip.reciprocal as the post-lowering signature.
// CHECK-LABEL: func.func @test_broadcast_div_to_mul_reciprocal
// CHECK-NOT: onnx.Div
// CHECK-NOT: hip.div
// CHECK: hip.reciprocal

// Large-Cin static 1x1 conv: rewritten to a hipBLASLt matmul plus a broadcast
// bias add. The key invariant is that NO hip.conv (MIOpen) survives — the
// pointwise conv is computed as a GEMM. The bias lowers to a broadcasting
// hip.add.
// CHECK-LABEL: func.func @test_pointwise_conv_large_cin_to_matmul
// CHECK-NOT: onnx.Conv
// CHECK-NOT: hip.conv
// CHECK-DAG: hip.matmul
// CHECK-DAG: hip.add

// Small-Cin static 1x1 conv: NOT rewritten to a matmul — it is let through to
// hip.conv (the fused-kernel routing happens later in HIP->LLVM).
// CHECK-LABEL: func.func @test_pointwise_conv_small_cin_to_conv
// CHECK-NOT: hip.matmul
// CHECK: hip.conv

// Dynamic-batch 1x1 conv: the rewrite is skipped, so it lowers via the normal
// MIOpen path and a hip.conv remains.
// CHECK-LABEL: func.func @test_pointwise_conv_dynamic_stays_conv
// CHECK-NOT: hip.matmul
// CHECK: hip.conv
