/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- hyper_connection_mixer.hpp - pre-mix gate path to one op -----------===//
//
// Collapse the gate path that feeds a hyper-connection pre-mix
//
//   %a = tensor.extract_slice %proj[.., 0] [.., K]
//   %s = hip.mul(%ctx) ins(%a, %scale)          (splat scalar constant)
//   %w = hip.swish(%ctx) ins(%s)
//   %m = hip.matmul_nbits(%ctx) ins(%w, %B, %S)  (4-bit, no zero points)
//   %y = hip.hyper_connection_pre_mix(%ctx) ins(%streams, %m)
//   %g = tensor.extract_slice %proj[.., K] [.., C]   (optional)
//
// into one hip.hyper_connection_mixer whose results replace %y and %g, so
// the runtime can run the decode case as one kernel.
//
// Before/After IR: hyper_connection_mixer.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateHyperConnectionMixerPattern(mlir::RewritePatternSet &patterns,
                                         mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
