/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- rope_layout_fusion.hpp - fp32 BNSH rope round trip to fp16 BSNH ----===//
//
// Fold transpose<BNSH>, cast<f32>, hip.rope, cast<f16>, transpose<BSNH>
// around an fp16 [B, S, N, D] tensor into one hip.rope on the tensor as is,
// rotating in fp32 with the fp32 caches.
//
// Before/After IR: rope_layout_fusion.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateRopeLayoutFusionPattern(mlir::RewritePatternSet &patterns,
                                     mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
