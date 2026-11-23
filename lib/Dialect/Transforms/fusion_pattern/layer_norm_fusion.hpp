/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- layer_norm_fusion.hpp - primitive LayerNorm chains to one hip op ---===//
//
// Collapse a last-axis LayerNorm written as primitive HIP ops into one
// hip.layer_norm, and a same-shape hip.add feeding that norm into one
// hip.skip_layer_norm. Every frontend that lowers to those HIP ops gets the
// fused kernel from this match. hip.layer_norm accumulates in fp32; leaving
// the chain as separate fp16 elementwise kernels overflows the squared
// deviations.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateLayerNormFusionPatterns(mlir::RewritePatternSet &patterns,
                                     mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
