// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @wrong_rank_controls(
    %ctx: !hip.context, %data: tensor<2x3xf32>, %valid: i1,
    %s: i64, %p: i64, %e: index) {
  %init = tensor.empty() : tensor<2x3xf32>
  // expected-error @+1 {{starts, steps, and extents must each contain exactly data-rank entries}}
  %r = hip.slice(%ctx) ins(%data : tensor<2x3xf32>)
      valid(%valid)
      starts(%s : i64)
      steps(%p : i64)
      extents(%e : index)
      outs(%init : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}

// -----

func.func @mismatched_dynamic_destination(
    %ctx: !hip.context, %data: tensor<?xf32>, %valid: i1,
    %s: i64, %p: i64, %e0: index, %e1: index) {
  %init = tensor.empty(%e0) : tensor<?xf32>
  // expected-error @+1 {{dynamic output dimension 0 must be allocated from the matching exact extent}}
  %r = hip.slice(%ctx) ins(%data : tensor<?xf32>)
      valid(%valid)
      starts(%s : i64)
      steps(%p : i64)
      extents(%e1 : index)
      outs(%init : tensor<?xf32>) : tensor<?xf32>
  return
}
