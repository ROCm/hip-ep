/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_ONNXTOHIPSR_ONNXTOHIPSR_H
#define HIP_CONVERSION_ONNXTOHIPSR_ONNXTOHIPSR_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace hipsr {

// Per-conversion declarations. createConvertOnnxToHipsrPass() is declared by
// GEN_PASS_DECL (defined by GEN_PASS_DEF in OnnxToHipsr.cpp).
// Pattern-population helpers for individual ONNX ops (added by follow-up
// layers) are declared here too. Registration lives in the aggregate
// hip/Conversion/Passes.h.
#define GEN_PASS_DECL_CONVERTONNXTOHIPSRPASS
#include "hip/Conversion/Passes.h.inc"

// Populates patterns that convert `onnx.Constant` into `hipsr.constant`
// (rank>=1 inline / mem_source / file_source) or `arith.constant` (rank-0
// scalar). Pure IR rewrite -- no file I/O and no size-threshold policy (that
// is layered on in the externalization pass).
void populateOnnxToHipsrConstantPatterns(::mlir::RewritePatternSet &patterns);

void populateCastConversionPatterns(::mlir::RewritePatternSet &patterns,
                                    ::mlir::MLIRContext *ctx);

void populateMatMulConversionPatterns(::mlir::RewritePatternSet &patterns,
                                      ::mlir::MLIRContext *ctx);

void populateExpandConversionPatterns(::mlir::RewritePatternSet &patterns,
                                      ::mlir::MLIRContext *ctx);

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIPSR_ONNXTOHIPSR_H
