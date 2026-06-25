// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Regression guard for the silent miscompile fixed by inserting
// `--empty-tensor-to-alloc-tensor` ahead of the pre-bufferize CSE in
// `buildOnnxToHipPipelineTail` (lib/Dialect/Transforms/Pipelines.cpp).
//
// Each HIP DPS op's `tensor.empty` is its private destination buffer.
// `tensor.empty` is side-effect-free, so a bare CSE merges all same-typed
// empties into ONE SSA value -- after bufferize, the distinct ops then share a
// single output buffer and clobber each other (cosine ~0.6 on whole encoders;
// the HIP DPS ops bufferize in-place without copy insertion). Converting the
// empties to `bufferization.alloc_tensor` first gives them allocation semantics
// (NOT memory-effect-free), so CSE leaves each destination distinct while still
// being free to dedup the side-effect-free shape/index arithmetic.
//
// HAZARD run: bare CSE collapses the two same-typed empties to one (the bug).
// RUN: hip-mlir-opt --cse %s | FileCheck %s --check-prefix=HAZARD
//
// FIX run: the production order keeps two distinct destinations.
// RUN: hip-mlir-opt --empty-tensor-to-alloc-tensor --cse %s | FileCheck %s --check-prefix=FIX

// Two independent matmuls (different operands, so CSE keeps both ops) each with
// its OWN same-typed `tensor.empty` destination.
func.func @two_dps_same_dest_type(%ctx: !hip.context,
                                  %a0: tensor<2x4xf16>, %b0: tensor<4x8xf16>,
                                  %a1: tensor<2x4xf16>, %b1: tensor<4x8xf16>)
    -> (tensor<2x8xf16>, tensor<2x8xf16>) {
  %e0 = tensor.empty() : tensor<2x8xf16>
  %y0 = hip.matmul(%ctx) ins(%a0, %b0 : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%e0 : tensor<2x8xf16>) : tensor<2x8xf16>
  %e1 = tensor.empty() : tensor<2x8xf16>
  %y1 = hip.matmul(%ctx) ins(%a1, %b1 : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%e1 : tensor<2x8xf16>) : tensor<2x8xf16>
  return %y0, %y1 : tensor<2x8xf16>, tensor<2x8xf16>
}

// Bare CSE merges the two empties into a single one (only one survives).
// HAZARD: tensor.empty() : tensor<2x8xf16>
// HAZARD-NOT: tensor.empty

// Each empty becomes its own alloc_tensor; CSE does not merge them, so the two
// matmuls keep distinct destinations (alloc, matmul, alloc, matmul).
// FIX: bufferization.alloc_tensor() : tensor<2x8xf16>
// FIX: hip.matmul
// FIX: bufferization.alloc_tensor() : tensor<2x8xf16>
// FIX: hip.matmul
