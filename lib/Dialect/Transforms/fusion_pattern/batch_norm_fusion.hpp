/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- batch_norm_fusion.hpp - elementwise batch norm to hip.batch_norm ---===//
//
// Collapse an inference batch normalization written out as primitive HIP ops
// into one hip.batch_norm:
//
//   y = (x - mean) * (1 / sqrt(variance + epsilon)) * scale + bias
//
// The scale and the bias are optional. Recovering hip.batch_norm lets
// hip-fold-batchnorm fold the normalization into a producing convolution. The
// pattern declines unless the per-channel parameters broadcast along dimension
// 1 of x, epsilon is known at compile time, and the full-size intermediates
// have a single value use.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateBatchNormFusionPattern(mlir::RewritePatternSet &patterns,
                                    mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
