/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- gelu_fusion.hpp - elementwise GELU chains to hip.gelu --------------===//
//
// Collapse a GELU written out as primitive HIP ops into one hip.gelu:
//
//   tanh form  0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//   erf form   0.5 * x * (1 + erf(x / sqrt(2)))      (x / sqrt(2) may also be
//                                                     x * (1 / sqrt(2)))
//
// Every frontend that lowers GELU to elementwise HIP ops gets the fused
// kernel from this one match. The pattern declines unless each intermediate
// has a single value use, every tensor has the type of x (no broadcasting),
// and the scalar constants match the formula.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateGeluFusionPattern(mlir::RewritePatternSet &patterns,
                               mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
