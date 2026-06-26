// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Regression guard for `--hip-dedup-dps-inits` (the liveness-safe replacement
// for the pre-bufferize CSE that caused the Whisper encoder aliasing
// miscompile). The pass must merge non-overlapping DPS init empties but keep
// simultaneously-live ones distinct.
//
// HAZARD run: a bare CSE merges ALL same-typed empties regardless of liveness.
// RUN: hip-mlir-opt --cse %s | FileCheck %s --check-prefix=HAZARD
//
// SAFE run: hip-dedup-dps-inits keeps simultaneously-live empties distinct.
// RUN: hip-mlir-opt --hip-dedup-dps-inits %s | FileCheck %s --check-prefix=SAFE

// Two independent matmuls whose results are BOTH returned (simultaneously live).
// The dedup pass must NOT merge their empties.
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

// Bare CSE merges the two empties into one (the aliasing bug).
// HAZARD: tensor.empty() : tensor<2x8xf16>
// HAZARD-NOT: tensor.empty

// hip-dedup-dps-inits keeps both (results are simultaneously live via return).
// SAFE: tensor.empty() : tensor<2x8xf16>
// SAFE: hip.matmul
// SAFE: tensor.empty() : tensor<2x8xf16>
// SAFE: hip.matmul
