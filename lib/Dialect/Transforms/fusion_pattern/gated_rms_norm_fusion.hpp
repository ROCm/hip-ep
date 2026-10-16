/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- gated_rms_norm_fusion.hpp - SiLU-gated RMS norm into one op --------===//
//
// Fold cast<f16>(cast<f32>(rms_norm(x)) * silu(cast<f32>(z))), the output
// gate of a gated-delta-net layer, into hip.gated_rms_norm. The norm may be
// per head (expand / rms_norm / collapse) or over the whole last axis; the
// SiLU product may be the primitive hip.sigmoid / hip.mul pair or
// hip.swiglu.
//
// Before/After IR: gated_rms_norm_fusion.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateGatedRmsNormFusionPattern(mlir::RewritePatternSet &patterns,
                                       mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
