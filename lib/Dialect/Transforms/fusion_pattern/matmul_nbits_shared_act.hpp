/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- matmul_nbits_shared_act.hpp - One int8 activation per W4A8 reader --===//
//
// Decoder layers project one activation through several int4 weights: q/k/v
// in attention, gate/up in the MLP. On the W4A8 dp4a path every
// hip.matmul_nbits quantizes that activation to int8 again before its GEMV.
// This pattern makes the int8 activation an IR value instead: one
// hip.quantize_act per (activation, block_size) read by two or more eligible
// projections, consumed through their `quantized(...)` operands.
//
// The quantized value is written for every row, so it does not depend on
// which kernel a consumer picks at runtime (M is dynamic); consumers off the
// dp4a path simply read the fp16 activation.
//
// Eligible projections: fp16 activation with a static K, bits = 4, no g_idx,
// symmetric or packed uint8 zero points, K % 32 == 0, block_size a power of
// two >= 32 that divides K.
//
// Before/After IR: matmul_nbits_shared_act.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateMatMulNBitsSharedActivationPatterns(
    mlir::RewritePatternSet &patterns, mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
