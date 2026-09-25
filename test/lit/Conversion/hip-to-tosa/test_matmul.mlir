// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// tosa.matmul takes rank-3 operands and will not broadcast a size-1 batch
// against a larger one, so hip.matmul's leading dims have to be folded away
// before it. Which dimension absorbs them depends on whether B is batched.
//
// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// An unbatched B is the same matrix for every batch element, so A's leading
// dims fold into M and the batch dimension stays 1.

// CHECK-LABEL: func.func @unbatched_b
// CHECK: %[[A:.*]] = tosa.reshape %arg1, {{.*}} -> tensor<1x36000x1280xf16>
// CHECK: %[[B:.*]] = tosa.reshape %arg2, {{.*}} -> tensor<1x1280x1280xf16>
// CHECK: %[[MM:.*]] = tosa.matmul %[[A]], %[[B]]
// CHECK-SAME: -> tensor<1x36000x1280xf16>
// CHECK: tosa.reshape %[[MM]], {{.*}} -> tensor<24x1500x1280xf16>
// CHECK-NOT: hip.matmul
func.func @unbatched_b(%ctx: !hip.context, %a: tensor<24x1500x1280xf16>,
                       %b: tensor<1280x1280xf16>,
                       %init: tensor<24x1500x1280xf16>)
    -> tensor<24x1500x1280xf16> attributes {rock.kernel} {
  %r = hip.matmul(%ctx) ins(%a, %b :
      tensor<24x1500x1280xf16>, tensor<1280x1280xf16>)
      outs(%init : tensor<24x1500x1280xf16>)
      {transA = 0 : i64, transB = 0 : i64} : tensor<24x1500x1280xf16>
  return %r : tensor<24x1500x1280xf16>
}

// -----

// Attention's Q@K^T: every (batch, head) pair has its own B, so folding the
// leading dims into M would be wrong. They collapse into tosa.matmul's batch
// dimension instead, which is legal here because both sides agree on them.

// CHECK-LABEL: func.func @batched_b
// CHECK: %[[A:.*]] = tosa.reshape %arg1, {{.*}} -> tensor<480x1500x64xf16>
// CHECK: %[[B:.*]] = tosa.reshape %arg2, {{.*}} -> tensor<480x64x1500xf16>
// CHECK: %[[MM:.*]] = tosa.matmul %[[A]], %[[B]]
// CHECK-SAME: -> tensor<480x1500x1500xf16>
// CHECK: tosa.reshape %[[MM]], {{.*}} -> tensor<24x20x1500x1500xf16>
// CHECK-NOT: hip.matmul
func.func @batched_b(%ctx: !hip.context, %a: tensor<24x20x1500x64xf16>,
                     %b: tensor<24x20x64x1500xf16>,
                     %init: tensor<24x20x1500x1500xf16>)
    -> tensor<24x20x1500x1500xf16> attributes {rock.kernel} {
  %r = hip.matmul(%ctx) ins(%a, %b :
      tensor<24x20x1500x64xf16>, tensor<24x20x64x1500xf16>)
      outs(%init : tensor<24x20x1500x1500xf16>)
      {transA = 0 : i64, transB = 0 : i64} : tensor<24x20x1500x1500xf16>
  return %r : tensor<24x20x1500x1500xf16>
}

// -----

// Attention's attn@V, the same shape with K and N swapped.

// CHECK-LABEL: func.func @batched_b_second
// CHECK: %[[A:.*]] = tosa.reshape %arg1, {{.*}} -> tensor<480x1500x1500xf16>
// CHECK: %[[B:.*]] = tosa.reshape %arg2, {{.*}} -> tensor<480x1500x64xf16>
// CHECK: %[[MM:.*]] = tosa.matmul %[[A]], %[[B]]
// CHECK-SAME: -> tensor<480x1500x64xf16>
// CHECK: tosa.reshape %[[MM]], {{.*}} -> tensor<24x20x1500x64xf16>
func.func @batched_b_second(%ctx: !hip.context,
                            %a: tensor<24x20x1500x1500xf16>,
                            %b: tensor<24x20x1500x64xf16>,
                            %init: tensor<24x20x1500x64xf16>)
    -> tensor<24x20x1500x64xf16> attributes {rock.kernel} {
  %r = hip.matmul(%ctx) ins(%a, %b :
      tensor<24x20x1500x1500xf16>, tensor<24x20x1500x64xf16>)
      outs(%init : tensor<24x20x1500x64xf16>)
      {transA = 0 : i64, transB = 0 : i64} : tensor<24x20x1500x64xf16>
  return %r : tensor<24x20x1500x64xf16>
}

// -----

// A rank-3 B is batched too -- nothing about the collapse is specific to
// rank 4.

// CHECK-LABEL: func.func @batched_b_rank3
// CHECK: %[[A:.*]] = tosa.reshape %arg1, {{.*}} -> tensor<8x4x16xf32>
// CHECK: %[[B:.*]] = tosa.reshape %arg2, {{.*}} -> tensor<8x16x32xf32>
// CHECK: tosa.matmul %[[A]], %[[B]]
// CHECK-SAME: -> tensor<8x4x32xf32>
func.func @batched_b_rank3(%ctx: !hip.context, %a: tensor<8x4x16xf32>,
                           %b: tensor<8x16x32xf32>, %init: tensor<8x4x32xf32>)
    -> tensor<8x4x32xf32> attributes {rock.kernel} {
  %r = hip.matmul(%ctx) ins(%a, %b : tensor<8x4x16xf32>, tensor<8x16x32xf32>)
      outs(%init : tensor<8x4x32xf32>)
      {transA = 0 : i64, transB = 0 : i64} : tensor<8x4x32xf32>
  return %r : tensor<8x4x32xf32>
}

// -----

// transB is attention's Q@K^T as the importer leaves it, having folded the
// onnx.Transpose into the flag. tosa.matmul has no such flag, so the
// transpose comes back as a real op on B's trailing two dims; the batch dims
// stay where they are.

// CHECK-LABEL: func.func @trans_b
// CHECK: %[[BT:.*]] = tosa.transpose %arg2 {perms = array<i32: 0, 1, 3, 2>}
// CHECK-SAME: -> tensor<1x12x64x577xf16>
// CHECK: %[[A:.*]] = tosa.reshape %arg1, {{.*}} -> tensor<12x577x64xf16>
// CHECK: %[[B:.*]] = tosa.reshape %[[BT]], {{.*}} -> tensor<12x64x577xf16>
// CHECK: %[[MM:.*]] = tosa.matmul %[[A]], %[[B]]
// CHECK-SAME: -> tensor<12x577x577xf16>
// CHECK: tosa.reshape %[[MM]], {{.*}} -> tensor<1x12x577x577xf16>
// CHECK-NOT: hip.matmul
func.func @trans_b(%ctx: !hip.context, %a: tensor<1x12x577x64xf16>,
                   %b: tensor<1x12x577x64xf16>,
                   %init: tensor<1x12x577x577xf16>)
    -> tensor<1x12x577x577xf16> attributes {rock.kernel} {
  %r = hip.matmul(%ctx) ins(%a, %b :
      tensor<1x12x577x64xf16>, tensor<1x12x577x64xf16>)
      outs(%init : tensor<1x12x577x577xf16>)
      {transA = 0 : i64, transB = 1 : i64} : tensor<1x12x577x577xf16>
  return %r : tensor<1x12x577x577xf16>
}

// -----

// transA on a rank-3 operand: K is the leading of the two trailing dims.

// CHECK-LABEL: func.func @trans_a_rank3
// CHECK: %[[AT:.*]] = tosa.transpose %arg1 {perms = array<i32: 0, 2, 1>}
// CHECK-SAME: -> tensor<8x4x16xf32>
// CHECK: %[[A:.*]] = tosa.reshape %[[AT]], {{.*}} -> tensor<8x4x16xf32>
// CHECK: %[[B:.*]] = tosa.reshape %arg2, {{.*}} -> tensor<8x16x32xf32>
// CHECK: tosa.matmul %[[A]], %[[B]]
// CHECK-SAME: -> tensor<8x4x32xf32>
func.func @trans_a_rank3(%ctx: !hip.context, %a: tensor<8x16x4xf32>,
                         %b: tensor<8x16x32xf32>, %init: tensor<8x4x32xf32>)
    -> tensor<8x4x32xf32> attributes {rock.kernel} {
  %r = hip.matmul(%ctx) ins(%a, %b : tensor<8x16x4xf32>, tensor<8x16x32xf32>)
      outs(%init : tensor<8x4x32xf32>)
      {transA = 1 : i64, transB = 0 : i64} : tensor<8x4x32xf32>
  return %r : tensor<8x4x32xf32>
}

// -----

// Both flags at once, against an unbatched B: A's leading dims still fold
// into M, which is valid because the transpose has already put M and K back
// in their logical order.

// CHECK-LABEL: func.func @trans_both_unbatched_b
// CHECK-DAG: %[[AT:.*]] = tosa.transpose %arg1 {perms = array<i32: 0, 2, 1>}
// CHECK-DAG: %[[BT:.*]] = tosa.transpose %arg2 {perms = array<i32: 1, 0>}
// CHECK: %[[A:.*]] = tosa.reshape %[[AT]], {{.*}} -> tensor<1x24x16xf32>
// CHECK: %[[B:.*]] = tosa.reshape %[[BT]], {{.*}} -> tensor<1x16x32xf32>
// CHECK: %[[MM:.*]] = tosa.matmul %[[A]], %[[B]]
// CHECK-SAME: -> tensor<1x24x32xf32>
// CHECK: tosa.reshape %[[MM]], {{.*}} -> tensor<8x3x32xf32>
// CHECK-NOT: hip.matmul
func.func @trans_both_unbatched_b(%ctx: !hip.context, %a: tensor<8x16x3xf32>,
                                  %b: tensor<32x16xf32>,
                                  %init: tensor<8x3x32xf32>)
    -> tensor<8x3x32xf32> attributes {rock.kernel} {
  %r = hip.matmul(%ctx) ins(%a, %b : tensor<8x16x3xf32>, tensor<32x16xf32>)
      outs(%init : tensor<8x3x32xf32>)
      {transA = 1 : i64, transB = 1 : i64} : tensor<8x3x32xf32>
  return %r : tensor<8x3x32xf32>
}

// -----

// A size-1 batch on B is the case tosa.matmul cannot express: it refuses to
// broadcast it up to A's, and folding A's batch into M would multiply every
// batch by the one B, which is a different computation. hip.matmul is
// unconditionally illegal here, so there is nothing to fall back to.
func.func @batch_mismatch(%ctx: !hip.context, %a: tensor<8x4x16xf32>,
                          %b: tensor<1x16x32xf32>, %init: tensor<8x4x32xf32>)
    -> tensor<8x4x32xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.matmul'}}
  %r = hip.matmul(%ctx) ins(%a, %b : tensor<8x4x16xf32>, tensor<1x16x32xf32>)
      outs(%init : tensor<8x4x32xf32>)
      {transA = 0 : i64, transB = 0 : i64} : tensor<8x4x32xf32>
  return %r : tensor<8x4x32xf32>
}
