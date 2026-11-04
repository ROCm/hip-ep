/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- gqa_rope_fold.hpp - fold hip.rope on Q/K into hip.gqa --------------===//
//
// Fold the two rotary embeddings that feed a hip.gqa's query and key
//
//   %q = hip.rope(%ctx) ins(%q_in, %pos, %cos, %sin)
//   %k = hip.rope(%ctx) ins(%k_in, %pos, %cos, %sin)
//   hip.gqa(%ctx) ins(%q, %k, %v, ...) {do_rotary = 0}
//
// into the attention op itself (do_rotary = 1 with the same position_ids,
// cos_cache and sin_cache), so the runtime applies RoPE inside its decode and
// prefill kernels instead of launching two extra kernels per layer. The
// pattern declines unless the rotation is exactly the one hip.gqa implements:
// half-rotated (interleaved = 0) over the whole head, positions taken from
// one position_ids tensor shared by Q and K, and each rope result consumed
// only by the attention op.
//
// Before/After IR: gqa_rope_fold.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateGqaRopeFoldPattern(mlir::RewritePatternSet &patterns,
                                mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
