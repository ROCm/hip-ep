/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- swiglu_fusion.hpp - hip.sigmoid + hip.mul chain to hip.swiglu ------===//
//
// Collapse the gated activation
//
//   %s = hip.sigmoid(%ctx) ins(%gate)
//   %a = hip.mul(%ctx) ins(%gate, %s)
//   %y = hip.mul(%ctx) ins(%a, %up)
//
// into one hip.swiglu. Both multiplies are commutative. The pattern declines
// unless the tensors are identical and the sigmoid and inner product each
// have a single use, because hip.swiglu is a flat elementwise kernel.
//
// Before/After IR: swiglu_fusion.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateSwigluFusionPattern(mlir::RewritePatternSet &patterns,
                                 mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
