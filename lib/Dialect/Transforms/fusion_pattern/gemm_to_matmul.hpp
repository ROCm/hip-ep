/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- gemm_to_matmul.hpp - hip.gemm to hip.matmul ------------------------===//
//
// A gemm that is a plain matmul plus a row bias (alpha = 1; no C, or beta = 1
// with a [N] / [1, N] C) becomes hip.matmul with the bias operand, so fusions
// and quant types are written against one op. Other gemms stay hip.gemm.
//
// Before/After IR: gemm_to_matmul.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateGemmToMatmulPattern(mlir::RewritePatternSet &patterns,
                                 mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
