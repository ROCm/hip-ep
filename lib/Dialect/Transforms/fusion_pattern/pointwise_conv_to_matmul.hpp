/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- pointwise_conv_to_matmul.hpp - 1x1 hip.conv as a batched matmul ----===//
//
// A `hip.conv` with a unit kernel, unit strides and dilations, no padding and a
// single group contracts each input pixel's channel vector against every
// filter, independently per pixel. In channels-first layout that is, per batch
// element n,
//
//   Y[n] (Cout x P) = W (Cout x Cin) . X[n] (Cin x P),   P = prod(spatial)
//
// a matmul whose left operand is shared by every batch element and whose right
// operand and result are the conv's own input and output, reshaped in place.
// The rewrite emits exactly that `hip.matmul`, plus a broadcast `hip.add` for
// the bias: no transposes and no data movement.
//
// The bias stays a separate op because a channels-first result puts the output
// channel on the GEMM's column axis, and hipBLASLt's bias epilogue only runs
// along the other one.
//
// What it leaves alone:
//  - a conv whose spatial extent is statically one pixel: patch-embed's
//    single-patch rewrite turns that into a 2-D `hip.gemm` with the bias in
//    the epilogue, which is strictly better;
//  - strided, padded, dilated and grouped convs, and integer or f64 ones,
//    which `hip.matmul` cannot run.
//
// Before/After IR: pointwise_conv_to_matmul.cpp.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populatePointwiseConvToMatmulPattern(mlir::RewritePatternSet &patterns,
                                          mlir::PatternBenefit benefit);

} // namespace fusion_transform
} // namespace hip
