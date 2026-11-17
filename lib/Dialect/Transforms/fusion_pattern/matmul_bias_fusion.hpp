/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- matmul_bias_fusion.hpp - hip.matmul + hip.add to matmul with bias --===//
//
// Fold a row-bias add into the matmul that feeds it:
//
//   %m = hip.matmul(%ctx) ins(%a, %w)
//   %y = hip.add(%ctx) ins(%m, %bias)        (either operand order)
//
// into one hip.matmul with the bias operand. The pattern reads only element
// types: a float chain and a chain of quant-typed tensors take the same path,
// and whether the fused op has a kernel is hip::supportsMatmulWithBias's
// answer (quantized types only today).
//
// For quantized types the intermediate matmul result's quantization is dropped:
// the fused kernel accumulates, adds the bias and quantizes once.
//
// Before/After IR: matmul_bias_fusion.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateMatmulBiasFusionPattern(mlir::RewritePatternSet &patterns,
                                     mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
