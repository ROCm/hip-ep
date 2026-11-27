/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- matmul_nbits_group.hpp - MatMulNBits ops sharing one activation ----===//
//
// Decoder layers project one normalized activation through several int4
// weights: q/k/v in attention, gate/up in the MLP. Each hip.matmul_nbits
// quantizes that activation again and launches its own GEMV, which for a
// single decode row is mostly launch overhead. This pattern replaces 2-4 such
// ops with one hip.matmul_nbits_group, which the runtime serves with one
// activation quantization and one GEMV launch.
//
// Members must be fp16 int4 projections without g_idx or bias, with the same
// K, block_size and scale type, and either all symmetric or all with packed
// uint8 zero points (AWQ-style asymmetric weights). Declining is always safe:
// the original ops are untouched and keep their own kernels.
//
// When the shared activation is the output of an all-fp16 hip.skip_rms_norm,
// hip.hyper_connection_pre_mix or flattened
// hip.hyper_connection_post_mix_norm, that kernel also emits the quantized
// row and the group reads it, removing the group's own quantization launch.
//
// A projection (or concat group member) whose output is only read through
// last-axis slices that tile it -- an exported model's fused projection split
// back into its parts -- becomes one group member per slice, reading a row
// range of the same weight. The slice of the full output is strided, and each
// one cost a separate copy launch before its consumer.
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
