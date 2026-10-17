/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- sigmoid_mul_fusion.hpp - x * sigmoid(gate) into hip.sigmoid_mul ----===//
//
// Fold an fp16 hip.mul of a tensor by the hip.sigmoid of a same-shaped gate
// into one hip.sigmoid_mul.
//
// Before/After IR: sigmoid_mul_fusion.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateSigmoidMulFusionPattern(mlir::RewritePatternSet &patterns,
                                     mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
