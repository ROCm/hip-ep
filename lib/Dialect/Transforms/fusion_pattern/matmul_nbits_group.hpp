/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- matmul_nbits_group.hpp - Projections of one int8 activation --------===//
//
// Once the shared-activation patterns have given a decoder layer's q/k/v (or
// gate/up) projections one int8 activation, the projections still launch one
// GEMV each, and at decode each launch is a fixed cost the GEMV body does not
// amortize. This pattern merges up to three of them into one
// hip.matmul_nbits_group, which the runtime runs as a single launch for one
// activation row and as the ordinary per-projection kernels otherwise.
//
// A gate/up pair whose only consumer is Mul(Mul(gate, Sigmoid(gate)), up)
// becomes one group with the silu_mul epilogue, so neither projection is
// written to memory on the single-launch path. The up projection's init stays
// on the op as `up_tmp`, which the multi-row path writes; memory planning
// allocates it like any other transient.
//
// Members: projections that read the same `quantized(...)` activation, int4,
// fp16 activation, output and scales, no g_idx and no bias, all symmetric or
// all with packed uint8 zero points, same K and block_size.
//
// Before/After IR: matmul_nbits_group.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateMatMulNBitsGroupPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
