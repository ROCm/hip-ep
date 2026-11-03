/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- qdq_chain_fusion.hpp - per-tensor Q/DQ chains to one kernel --------===//
//
// Two rewrites for graphs whose activations carry per-tensor QDQ between
// float ops. Both are bit-exact: every rounding and saturation of the
// unfused chain is kept, only the integer intermediates leave memory.
//
//   * quantized SwiGLU: Q(gate), Q(up), qsigmoid, two qmul and the trailing
//     DQ become hip.qswiglu;
//   * a Q read only by DQs with the same parameters becomes one
//     hip.fake_quantize.
//
// They must run after the PDLL Q/DQ fusions have formed hip.qmul and
// hip.qsigmoid, and the round-trip rewrite must not preempt a fusion that
// consumes a DQ (hip.qmatmul, hip.qmul), so `run()` applies them in a second
// greedy phase.
//
// Before/After IR: qdq_chain_fusion.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateQdqChainFusionPatterns(mlir::RewritePatternSet &patterns);

} // namespace fusion_transform
} // namespace hip
