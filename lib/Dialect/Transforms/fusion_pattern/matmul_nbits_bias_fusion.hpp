/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- matmul_nbits_bias_fusion.hpp - fold a bias add into MatMulNBits ----===//
//
// Fold a per-output-column add into the optional bias operand of
// hip.matmul_nbits:
//
//   %y = hip.matmul_nbits(%ctx) ins(%a, %b, %scales) outs(%init)
//   %z = hip.add(%ctx) ins(%y, %bias) outs(%init2)      // either operand order
//     ->
//   %z = hip.matmul_nbits(%ctx) ins(%a, %b, %scales) bias(%bias) outs(%init2)
//
// Exporters that keep MatMulNBits bias-free emit the add as its own kernel,
// one extra launch and one extra pass over the output per projection. The
// pattern declines unless the matmul has no bias yet, its result has the add
// as its single value use, the bias is a rank-1 [N] tensor of the output
// element type, and the add result has the matmul's type (no broadcast).
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateMatMulNBitsBiasFusionPattern(mlir::RewritePatternSet &patterns,
                                          mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
