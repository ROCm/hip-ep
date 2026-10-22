// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --canonicalize %s | FileCheck %s

// What this file tests
// --------------------
// `TransposeOp::getCanonicalizationPatterns` -- collapsing chained Transposes.
// See lib/Dialect/IR/HipTransposeCanonicalize.cpp.
//
// A permutation composed with its inverse is a copy, but each op in the pair
// verifies on its own, so nothing that reasons one op at a time removes it. A
// decoder carrying a [1,32,4096,96] f16 KV cache pays 25 MB of reads and writes
// per token per Transpose to reproduce its input exactly.
//
// The negative cases are the point of the guards: a chain whose intermediate
// has another reader must not be rewritten into a third permutation, and a
// permutation pair that does not actually compose to the identity must not be
// substituted away, because doing so hands the consumer a different shape.

module {
  // --------------------------------------------------------------------------
  // Positive: [0,2,1,3] is its own inverse, so the pair is the identity and the
  // second Transpose is replaced by the first one's input. Both ops go.
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @fold_inverse_pair
  // CHECK-NOT:     hip.transpose
  // CHECK:         return %arg1
  func.func @fold_inverse_pair(
      %ctx: !hip.context,
      %kv: tensor<1x32x4096x96xf16>) -> tensor<1x32x4096x96xf16> {
    %bsnh = tensor.empty() : tensor<1x4096x32x96xf16>
    %0 = hip.transpose(%ctx) ins(%kv : tensor<1x32x4096x96xf16>)
                             outs(%bsnh : tensor<1x4096x32x96xf16>)
                             {perm = [0, 2, 1, 3]} : tensor<1x4096x32x96xf16>
    %bnsh = tensor.empty() : tensor<1x32x4096x96xf16>
    %1 = hip.transpose(%ctx) ins(%0 : tensor<1x4096x32x96xf16>)
                             outs(%bnsh : tensor<1x32x4096x96xf16>)
                             {perm = [0, 2, 1, 3]} : tensor<1x32x4096x96xf16>
    return %1 : tensor<1x32x4096x96xf16>
  }

  // --------------------------------------------------------------------------
  // Positive: two permutations that do not undo each other still collapse to
  // one. [1,2,0] after [2,0,1] composes to the identity of neither, so the
  // result is a single Transpose carrying the composed permutation.
  //
  // first = [2,0,1], second = [1,2,0]  ->  composed[i] = first[second[i]]
  //                                     =  [first[1], first[2], first[0]]
  //                                    =  [0, 1, 2]
  // which is the identity, so this pair also disappears entirely.
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @fold_rotation_pair
  // CHECK-NOT:     hip.transpose
  // CHECK:         return %arg1
  func.func @fold_rotation_pair(
      %ctx: !hip.context,
      %x: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %a = tensor.empty() : tensor<4x2x3xf32>
    %0 = hip.transpose(%ctx) ins(%x : tensor<2x3x4xf32>)
                             outs(%a : tensor<4x2x3xf32>)
                             {perm = [2, 0, 1]} : tensor<4x2x3xf32>
    %b = tensor.empty() : tensor<2x3x4xf32>
    %1 = hip.transpose(%ctx) ins(%0 : tensor<4x2x3xf32>)
                             outs(%b : tensor<2x3x4xf32>)
                             {perm = [1, 2, 0]} : tensor<2x3x4xf32>
    return %1 : tensor<2x3x4xf32>
  }

  // --------------------------------------------------------------------------
  // Positive: a chain that composes to a real permutation becomes one
  // Transpose rather than two.
  //
  // first = [1,0,2], second = [0,2,1]  ->  [first[0], first[2], first[1]]
  //                                    =  [1, 2, 0]
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @compose_chain
  // CHECK:         hip.transpose
  // CHECK-SAME:      perm = [1, 2, 0]
  // CHECK-NOT:     hip.transpose
  func.func @compose_chain(
      %ctx: !hip.context,
      %x: tensor<2x3x4xf32>) -> tensor<3x4x2xf32> {
    %a = tensor.empty() : tensor<3x2x4xf32>
    %0 = hip.transpose(%ctx) ins(%x : tensor<2x3x4xf32>)
                             outs(%a : tensor<3x2x4xf32>)
                             {perm = [1, 0, 2]} : tensor<3x2x4xf32>
    %b = tensor.empty() : tensor<3x4x2xf32>
    %1 = hip.transpose(%ctx) ins(%0 : tensor<3x2x4xf32>)
                             outs(%b : tensor<3x4x2xf32>)
                             {perm = [0, 2, 1]} : tensor<3x4x2xf32>
    return %1 : tensor<3x4x2xf32>
  }

  // --------------------------------------------------------------------------
  // Positive: a permutation that permutes nothing is a copy on its own.
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @fold_identity
  // CHECK-NOT:     hip.transpose
  // CHECK:         return %arg1
  func.func @fold_identity(
      %ctx: !hip.context,
      %x: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %a = tensor.empty() : tensor<2x3x4xf32>
    %0 = hip.transpose(%ctx) ins(%x : tensor<2x3x4xf32>)
                             outs(%a : tensor<2x3x4xf32>)
                             {perm = [0, 1, 2]} : tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }

  // --------------------------------------------------------------------------
  // Negative: the intermediate is returned as well, so the first Transpose
  // keeps running. The second one still collapses -- the pair is the identity
  // and substituting the source removes a permutation without adding one --
  // but the first must survive for its other reader.
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @keep_first_for_other_reader
  // CHECK:         %[[MID:.*]] = hip.transpose
  // CHECK-SAME:      perm = [0, 2, 1, 3]
  // CHECK-NOT:     hip.transpose
  // CHECK:         return %arg1, %[[MID]]
  func.func @keep_first_for_other_reader(
      %ctx: !hip.context,
      %kv: tensor<1x32x4096x96xf16>)
      -> (tensor<1x32x4096x96xf16>, tensor<1x4096x32x96xf16>) {
    %bsnh = tensor.empty() : tensor<1x4096x32x96xf16>
    %0 = hip.transpose(%ctx) ins(%kv : tensor<1x32x4096x96xf16>)
                             outs(%bsnh : tensor<1x4096x32x96xf16>)
                             {perm = [0, 2, 1, 3]} : tensor<1x4096x32x96xf16>
    %bnsh = tensor.empty() : tensor<1x32x4096x96xf16>
    %1 = hip.transpose(%ctx) ins(%0 : tensor<1x4096x32x96xf16>)
                             outs(%bnsh : tensor<1x32x4096x96xf16>)
                             {perm = [0, 2, 1, 3]} : tensor<1x32x4096x96xf16>
    return %1, %0 : tensor<1x32x4096x96xf16>, tensor<1x4096x32x96xf16>
  }

  // --------------------------------------------------------------------------
  // Negative: a non-identity chain whose intermediate has another reader.
  // Rewriting to the composed permutation would leave three Transposes where
  // there were two, so both are left alone.
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @keep_chain_with_other_reader
  // CHECK:         hip.transpose
  // CHECK-SAME:      perm = [1, 0, 2]
  // CHECK:         hip.transpose
  // CHECK-SAME:      perm = [0, 2, 1]
  func.func @keep_chain_with_other_reader(
      %ctx: !hip.context,
      %x: tensor<2x3x4xf32>)
      -> (tensor<3x4x2xf32>, tensor<3x2x4xf32>) {
    %a = tensor.empty() : tensor<3x2x4xf32>
    %0 = hip.transpose(%ctx) ins(%x : tensor<2x3x4xf32>)
                             outs(%a : tensor<3x2x4xf32>)
                             {perm = [1, 0, 2]} : tensor<3x2x4xf32>
    %b = tensor.empty() : tensor<3x4x2xf32>
    %1 = hip.transpose(%ctx) ins(%0 : tensor<3x2x4xf32>)
                             outs(%b : tensor<3x4x2xf32>)
                             {perm = [0, 2, 1]} : tensor<3x4x2xf32>
    return %1, %0 : tensor<3x4x2xf32>, tensor<3x2x4xf32>
  }

  // --------------------------------------------------------------------------
  // Negative: a lone Transpose that really does permute stays.
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @keep_single_transpose
  // CHECK:         hip.transpose
  // CHECK-SAME:      perm = [0, 2, 1, 3]
  func.func @keep_single_transpose(
      %ctx: !hip.context,
      %kv: tensor<1x32x4096x96xf16>) -> tensor<1x4096x32x96xf16> {
    %bsnh = tensor.empty() : tensor<1x4096x32x96xf16>
    %0 = hip.transpose(%ctx) ins(%kv : tensor<1x32x4096x96xf16>)
                             outs(%bsnh : tensor<1x4096x32x96xf16>)
                             {perm = [0, 2, 1, 3]} : tensor<1x4096x32x96xf16>
    return %0 : tensor<1x4096x32x96xf16>
  }
}
