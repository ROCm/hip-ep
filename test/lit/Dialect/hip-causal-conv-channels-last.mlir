// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --canonicalize %s | FileCheck %s

// What this file tests
// --------------------
// `CausalConvWithStateOp::getCanonicalizationPatterns` -- the fold that absorbs
// a [0,2,1] Transpose pair around a causal convolution into its `channels_last`
// attribute. See lib/Dialect/IR/HipCausalConvCanonicalize.cpp.
//
// The exporter emits Transpose/Conv/Transpose because ONNX Conv is
// channels-first, not because the kernel needs that layout. Both transposes are
// pure data movement, and on Qwen3.6-35B-A3B the pair costs more than the
// convolution between them.
//
// The negative cases matter as much as the positive one: each guard exists
// because folding without it produces IR that still verifies. A transpose with
// a second reader keeps running, so folding buys the attribute without removing
// the traffic; a permutation that is not the last-two-axis swap means the
// buffer would be reinterpreted rather than relabelled.

module {
  // --------------------------------------------------------------------------
  // Positive: the whole triple collapses to one convolution. Both transposes
  // go away -- the leading one becomes dead, the trailing one is replaced by
  // the convolution's own result.
  // --------------------------------------------------------------------------
  func.func @fold_transpose_pair(
      %ctx: !hip.context,
      %x: tensor<1x128x64xf16>,
      %w: tensor<64x1x4xf16>,
      %b: tensor<64xf16>,
      %past: tensor<1x64x3xf16>)
      -> (tensor<1x128x64xf16>, tensor<1x64x3xf16>) {
    // CHECK-LABEL: func.func @fold_transpose_pair
    // CHECK-NOT: hip.transpose
    // CHECK: hip.causal_conv_with_state
    // CHECK-SAME: channels_last = true
    // CHECK-NOT: hip.transpose

    %e0 = tensor.empty() : tensor<1x64x128xf16>
    %t0 = hip.transpose(%ctx) ins(%x : tensor<1x128x64xf16>)
        outs(%e0 : tensor<1x64x128xf16>) {perm = [0, 2, 1]}
        : tensor<1x64x128xf16>

    %e1 = tensor.empty() : tensor<1x64x128xf16>
    %e2 = tensor.empty() : tensor<1x64x3xf16>
    %y, %s = hip.causal_conv_with_state(%ctx)
        ins(%t0, %w, %b, %past :
            tensor<1x64x128xf16>, tensor<64x1x4xf16>,
            tensor<64xf16>, tensor<1x64x3xf16>)
        outs(%e1, %e2 : tensor<1x64x128xf16>, tensor<1x64x3xf16>)
        {activation = "silu", ndim = 1 : i64}
        : tensor<1x64x128xf16>, tensor<1x64x3xf16>

    %e3 = tensor.empty() : tensor<1x128x64xf16>
    %z = hip.transpose(%ctx) ins(%y : tensor<1x64x128xf16>)
        outs(%e3 : tensor<1x128x64xf16>) {perm = [0, 2, 1]}
        : tensor<1x128x64xf16>

    return %z, %s : tensor<1x128x64xf16>, tensor<1x64x3xf16>
  }

  // --------------------------------------------------------------------------
  // Negative: the convolution's output is not consumed by a transpose, so
  // there is no channels-last output buffer to write into.
  // --------------------------------------------------------------------------
  func.func @no_fold_without_trailing_transpose(
      %ctx: !hip.context,
      %x: tensor<1x128x64xf16>,
      %w: tensor<64x1x4xf16>)
      -> (tensor<1x64x128xf16>, tensor<1x64x3xf16>) {
    // CHECK-LABEL: func.func @no_fold_without_trailing_transpose
    // CHECK: hip.transpose
    // CHECK-NOT: channels_last

    %e0 = tensor.empty() : tensor<1x64x128xf16>
    %t0 = hip.transpose(%ctx) ins(%x : tensor<1x128x64xf16>)
        outs(%e0 : tensor<1x64x128xf16>) {perm = [0, 2, 1]}
        : tensor<1x64x128xf16>

    %e1 = tensor.empty() : tensor<1x64x128xf16>
    %e2 = tensor.empty() : tensor<1x64x3xf16>
    %y, %s = hip.causal_conv_with_state(%ctx)
        ins(%t0, %w : tensor<1x64x128xf16>, tensor<64x1x4xf16>)
        outs(%e1, %e2 : tensor<1x64x128xf16>, tensor<1x64x3xf16>)
        {ndim = 1 : i64}
        : tensor<1x64x128xf16>, tensor<1x64x3xf16>

    return %y, %s : tensor<1x64x128xf16>, tensor<1x64x3xf16>
  }

  // --------------------------------------------------------------------------
  // Negative: the leading transpose has a second reader, so it survives the
  // fold. Rewriting the convolution would leave both the transpose and a
  // layout change, which is strictly worse than leaving it alone.
  // --------------------------------------------------------------------------
  func.func @no_fold_when_transpose_is_reused(
      %ctx: !hip.context,
      %x: tensor<1x128x64xf16>,
      %w: tensor<64x1x4xf16>)
      -> (tensor<1x128x64xf16>, tensor<1x64x3xf16>, tensor<1x64x128xf16>) {
    // CHECK-LABEL: func.func @no_fold_when_transpose_is_reused
    // CHECK: hip.transpose
    // CHECK-NOT: channels_last

    %e0 = tensor.empty() : tensor<1x64x128xf16>
    %t0 = hip.transpose(%ctx) ins(%x : tensor<1x128x64xf16>)
        outs(%e0 : tensor<1x64x128xf16>) {perm = [0, 2, 1]}
        : tensor<1x64x128xf16>

    %e1 = tensor.empty() : tensor<1x64x128xf16>
    %e2 = tensor.empty() : tensor<1x64x3xf16>
    %y, %s = hip.causal_conv_with_state(%ctx)
        ins(%t0, %w : tensor<1x64x128xf16>, tensor<64x1x4xf16>)
        outs(%e1, %e2 : tensor<1x64x128xf16>, tensor<1x64x3xf16>)
        {ndim = 1 : i64}
        : tensor<1x64x128xf16>, tensor<1x64x3xf16>

    %e3 = tensor.empty() : tensor<1x128x64xf16>
    %z = hip.transpose(%ctx) ins(%y : tensor<1x64x128xf16>)
        outs(%e3 : tensor<1x128x64xf16>) {perm = [0, 2, 1]}
        : tensor<1x128x64xf16>

    return %z, %s, %t0
        : tensor<1x128x64xf16>, tensor<1x64x3xf16>, tensor<1x64x128xf16>
  }

  // --------------------------------------------------------------------------
  // Negative: a permutation that is not the last-two-axis swap. The pair still
  // composes to the identity, but the convolution's channels axis is not where
  // channels_last says it is.
  // --------------------------------------------------------------------------
  func.func @no_fold_wrong_permutation(
      %ctx: !hip.context,
      %x: tensor<128x1x64xf16>,
      %w: tensor<128x1x4xf16>)
      -> (tensor<128x1x64xf16>, tensor<1x128x3xf16>) {
    // CHECK-LABEL: func.func @no_fold_wrong_permutation
    // CHECK: hip.transpose
    // CHECK-NOT: channels_last

    %e0 = tensor.empty() : tensor<1x128x64xf16>
    %t0 = hip.transpose(%ctx) ins(%x : tensor<128x1x64xf16>)
        outs(%e0 : tensor<1x128x64xf16>) {perm = [1, 0, 2]}
        : tensor<1x128x64xf16>

    %e1 = tensor.empty() : tensor<1x128x64xf16>
    %e2 = tensor.empty() : tensor<1x128x3xf16>
    %y, %s = hip.causal_conv_with_state(%ctx)
        ins(%t0, %w : tensor<1x128x64xf16>, tensor<128x1x4xf16>)
        outs(%e1, %e2 : tensor<1x128x64xf16>, tensor<1x128x3xf16>)
        {ndim = 1 : i64}
        : tensor<1x128x64xf16>, tensor<1x128x3xf16>

    %e3 = tensor.empty() : tensor<128x1x64xf16>
    %z = hip.transpose(%ctx) ins(%y : tensor<1x128x64xf16>)
        outs(%e3 : tensor<128x1x64xf16>) {perm = [1, 0, 2]}
        : tensor<128x1x64xf16>

    return %z, %s : tensor<128x1x64xf16>, tensor<1x128x3xf16>
  }

  // --------------------------------------------------------------------------
  // Positive: k=12 is past the templated kernels' 8 taps, but the channels-last
  // kernel carries wider filters on its dynamic-K path, so the fold applies.
  // --------------------------------------------------------------------------
  func.func @fold_past_templated_kernel_width(
      %ctx: !hip.context,
      %x: tensor<1x128x64xf16>,
      %w: tensor<64x1x12xf16>)
      -> (tensor<1x128x64xf16>, tensor<1x64x11xf16>) {
    // CHECK-LABEL: func.func @fold_past_templated_kernel_width
    // CHECK: channels_last
    // CHECK-NOT: hip.transpose

    %e0 = tensor.empty() : tensor<1x64x128xf16>
    %t0 = hip.transpose(%ctx) ins(%x : tensor<1x128x64xf16>)
        outs(%e0 : tensor<1x64x128xf16>) {perm = [0, 2, 1]}
        : tensor<1x64x128xf16>

    %e1 = tensor.empty() : tensor<1x64x128xf16>
    %e2 = tensor.empty() : tensor<1x64x11xf16>
    %y, %s = hip.causal_conv_with_state(%ctx)
        ins(%t0, %w : tensor<1x64x128xf16>, tensor<64x1x12xf16>)
        outs(%e1, %e2 : tensor<1x64x128xf16>, tensor<1x64x11xf16>)
        {ndim = 1 : i64}
        : tensor<1x64x128xf16>, tensor<1x64x11xf16>

    %e3 = tensor.empty() : tensor<1x128x64xf16>
    %z = hip.transpose(%ctx) ins(%y : tensor<1x64x128xf16>)
        outs(%e3 : tensor<1x128x64xf16>) {perm = [0, 2, 1]}
        : tensor<1x128x64xf16>

    return %z, %s : tensor<1x128x64xf16>, tensor<1x64x11xf16>
  }

  // --------------------------------------------------------------------------
  // Negative: k=200 is past the width the fold will commit to. The kernel's own
  // ceiling is set by how much LDS a block can hold for the per-lane window,
  // which is not knowable at compile time, so the fold stops well short of it
  // rather than risk converting a working convolution into a runtime failure.
  // --------------------------------------------------------------------------
  func.func @no_fold_when_kernel_too_wide(
      %ctx: !hip.context,
      %x: tensor<1x128x64xf16>,
      %w: tensor<64x1x200xf16>)
      -> (tensor<1x128x64xf16>, tensor<1x64x199xf16>) {
    // CHECK-LABEL: func.func @no_fold_when_kernel_too_wide
    // CHECK: hip.transpose
    // CHECK-NOT: channels_last

    %e0 = tensor.empty() : tensor<1x64x128xf16>
    %t0 = hip.transpose(%ctx) ins(%x : tensor<1x128x64xf16>)
        outs(%e0 : tensor<1x64x128xf16>) {perm = [0, 2, 1]}
        : tensor<1x64x128xf16>

    %e1 = tensor.empty() : tensor<1x64x128xf16>
    %e2 = tensor.empty() : tensor<1x64x199xf16>
    %y, %s = hip.causal_conv_with_state(%ctx)
        ins(%t0, %w : tensor<1x64x128xf16>, tensor<64x1x200xf16>)
        outs(%e1, %e2 : tensor<1x64x128xf16>, tensor<1x64x199xf16>)
        {ndim = 1 : i64}
        : tensor<1x64x128xf16>, tensor<1x64x199xf16>

    %e3 = tensor.empty() : tensor<1x128x64xf16>
    %z = hip.transpose(%ctx) ins(%y : tensor<1x64x128xf16>)
        outs(%e3 : tensor<1x128x64xf16>) {perm = [0, 2, 1]}
        : tensor<1x128x64xf16>

    return %z, %s : tensor<1x128x64xf16>, tensor<1x64x199xf16>
  }

  // --------------------------------------------------------------------------
  // Negative: already folded. A pattern that re-matched its own output would
  // rewrite forever, so the canonicalizer completing at all is the assertion
  // here. The transposes stay because this IR is deliberately inconsistent --
  // a real graph has none left by this point.
  // --------------------------------------------------------------------------
  func.func @no_refold_when_already_channels_last(
      %ctx: !hip.context,
      %x: tensor<1x64x128xf16>,
      %w: tensor<64x1x4xf16>)
      -> (tensor<1x64x128xf16>, tensor<1x64x3xf16>) {
    // CHECK-LABEL: func.func @no_refold_when_already_channels_last
    // CHECK: hip.causal_conv_with_state
    // CHECK-SAME: channels_last = true

    %e0 = tensor.empty() : tensor<1x128x64xf16>
    %t0 = hip.transpose(%ctx) ins(%x : tensor<1x64x128xf16>)
        outs(%e0 : tensor<1x128x64xf16>) {perm = [0, 2, 1]}
        : tensor<1x128x64xf16>

    %e1 = tensor.empty() : tensor<1x128x64xf16>
    %e2 = tensor.empty() : tensor<1x64x3xf16>
    %y, %s = hip.causal_conv_with_state(%ctx)
        ins(%t0, %w : tensor<1x128x64xf16>, tensor<64x1x4xf16>)
        outs(%e1, %e2 : tensor<1x128x64xf16>, tensor<1x64x3xf16>)
        {ndim = 1 : i64, channels_last = true}
        : tensor<1x128x64xf16>, tensor<1x64x3xf16>

    %e3 = tensor.empty() : tensor<1x64x128xf16>
    %z = hip.transpose(%ctx) ins(%y : tensor<1x128x64xf16>)
        outs(%e3 : tensor<1x64x128xf16>) {perm = [0, 2, 1]}
        : tensor<1x64x128xf16>

    return %z, %s : tensor<1x64x128xf16>, tensor<1x64x3xf16>
  }
}
