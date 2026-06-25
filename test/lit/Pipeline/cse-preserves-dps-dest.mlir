// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Regression guard for the silent miscompile that a `cse` before bufferize
// causes in `buildOnnxToHipPipelineTail` (lib/Dialect/Transforms/Pipelines.cpp).
//
// Each HIP DPS op's `tensor.empty` is its private destination buffer.
// `tensor.empty` is side-effect-free, so a bare CSE merges all same-typed
// empties into ONE SSA value -- after bufferize the distinct ops then share a
// single output buffer and clobber each other (cosine ~0.6 on whole encoders;
// the HIP DPS ops bufferize in-place without copy insertion). The pipeline tail
// therefore runs ONLY `--canonicalize` after `--hip-infer-shapes` (to fold the
// now-static `tensor.dim` arithmetic) and deliberately NO `--cse`: canonicalize
// folds/DCEs without coalescing distinct `tensor.empty` destinations, and the
// liveness-aware `buffer-deallocation` / `--hip-pool-allocs` reuse buffers
// correctly downstream. (An earlier fix ran `--empty-tensor-to-alloc-tensor`
// to make a pre-bufferize CSE safe, but `alloc_tensor` disables empty-tensor-
// elimination and ~4x-ed the GPU pool on some models, so it was dropped.)
//
// HAZARD run: a bare CSE collapses the two same-typed empties to one (the bug
// we must not reintroduce before bufferize).
// RUN: hip-mlir-opt --cse %s | FileCheck %s --check-prefix=HAZARD
//
// OK run: the pass the pipeline actually uses keeps the two destinations distinct.
// RUN: hip-mlir-opt --canonicalize %s | FileCheck %s --check-prefix=OK

// Two independent matmuls (different operands) each with its OWN same-typed
// `tensor.empty` destination.
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

// Bare CSE merges the two empties into a single one (only one survives) -- this
// is why the pipeline must NOT run cse before bufferize.
// HAZARD: tensor.empty() : tensor<2x8xf16>
// HAZARD-NOT: tensor.empty

// canonicalize keeps both empties distinct, so each matmul retains its own
// destination (empty, matmul, empty, matmul).
// OK: tensor.empty() : tensor<2x8xf16>
// OK: hip.matmul
// OK: tensor.empty() : tensor<2x8xf16>
// OK: hip.matmul
