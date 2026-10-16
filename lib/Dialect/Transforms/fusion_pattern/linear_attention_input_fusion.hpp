/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- linear_attention_input_fusion.hpp - LA input prep into the op ------===//
//
// Fold the elementwise preparation a gated-delta-net layer applies to its
// projections into hip.linear_attention, which runs it in one launch:
//
//   - q/k per-head RMS norms (one weight) and a splat q scale
//                                     -> qk_norm(weight) / qk_norm_epsilon /
//                                        q_scale
//   - q, k, v slices of one tensor    -> packed QKV (key/value absent)
//   - cast(scale * softplus(cast(a) + bias)) decay
//                                     -> decay_softplus(bias, scale)
//   - sigmoid(b) beta                 -> beta_sigmoid
//
// Before/After IR: linear_attention_input_fusion.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateLinearAttentionInputFusionPatterns(
    mlir::RewritePatternSet &patterns, mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
