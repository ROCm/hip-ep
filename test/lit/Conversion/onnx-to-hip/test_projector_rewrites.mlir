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
