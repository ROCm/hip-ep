/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- mla_attention_fusion.hpp - Expanded latent cache to MLA attention --===//
//
// Rewrite a hip.kv_attention whose keys and values are up-projections of a
// latent cache [B, 1, T, L + R] (keys [c @ W_k, k_r] with k_r broadcast over
// the heads, values c @ W_v) into one hip.mla_attention on the cache. The
// pattern declines unless every slice provably covers its whole axis and the
// up-projection weights are constants.
//
// Before/After IR: mla_attention_fusion.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateMlaAttentionFusionPattern(mlir::RewritePatternSet &patterns,
                                       mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
