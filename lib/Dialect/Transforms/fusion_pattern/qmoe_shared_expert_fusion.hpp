/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- qmoe_shared_expert_fusion.hpp - shared expert into hip.qmoe --------===//
//
// Fold a Qwen-MoE style shared expert into the routed hip.qmoe beside it:
//
//   %g = hip.matmul_nbits(%x, Wg)     %u = hip.matmul_nbits(%x, Wu)
//   %a = hip.swiglu(%g, %u)           %d = hip.matmul_nbits(%a, Wd)
//   %l = hip.matmul_nbits(%x, Wl)     (N = 1)
//   %y = hip.add(hip.qmoe(%x, ...), hip.mul(%d, hip.sigmoid(%l)))
//
// becomes one hip.qmoe carrying the shared expert's weights and the logit %l.
// The logit projection stays a hip.matmul_nbits. Both commutative ops may
// list their operands either way. The pattern runs on the hip.swiglu that
// PR #1123's fusion produces, so it needs no SiLU matching of its own.
//
// Before/After IR: qmoe_shared_expert_fusion.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateQMoESharedExpertFusionPattern(mlir::RewritePatternSet &patterns,
                                           mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
