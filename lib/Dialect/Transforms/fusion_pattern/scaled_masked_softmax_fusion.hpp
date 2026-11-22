/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- scaled_masked_softmax_fusion.hpp - scale, id mask, softmax ---------===//
//
// Collapse attention-score post-processing
//
//   %s = hip.mul(%ctx) ins(%scores, %scale)            scalar constant scale
//   %m = hip.equal(%ctx) ins(%rows[R,1], %cols[1,C])   1-D ids, expanded
//   %w = hip.where(%ctx) ins(%m, %s, %fill)            scalar constant fill
//   %p = hip.miopen.softmax(%ctx) ins(%w)
//
// into one hip.scaled_masked_softmax that reads the two id vectors instead of
// the R x C mask, so the scaled and masked score tensors are never written.
// Vision encoders build this mask to keep attention inside a video frame.
//
// Before/After IR: scaled_masked_softmax_fusion.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateScaledMaskedSoftmaxFusionPattern(mlir::RewritePatternSet &patterns,
                                              mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
