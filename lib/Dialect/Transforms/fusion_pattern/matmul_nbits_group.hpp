/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- matmul_nbits_group.hpp - MatMulNBits ops sharing one activation ----===//
//
// Decoder layers project one normalized activation through several int4
// weights: q/k/v in attention, gate/up in the MLP. Each hip.matmul_nbits
// quantizes that activation again and launches its own GEMV, which for a
// single decode row is mostly launch overhead. This pattern replaces 2-3 such
// ops with one hip.matmul_nbits_group, which the runtime serves with one
// activation quantization and one GEMV launch.
//
// Members must be fp16 int4 projections without g_idx or bias, with the same
// K, block_size and scale type, and either all symmetric or all with packed
// uint8 zero points (AWQ-style asymmetric weights). Declining is always safe:
// the original ops are untouched and keep their own kernels.
//
// When the shared activation is a hip.skip_rms_norm output, the norm also
// emits the quantized row and the group reads it, removing the group's own
// quantization launch.
//
// Before/After IR: matmul_nbits_group.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateMatMulNBitsGroupPattern(mlir::RewritePatternSet &patterns,
                                     mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
