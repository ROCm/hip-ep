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

void populateOnnxToHipsrConstantPatterns(::mlir::RewritePatternSet &patterns);

void populateCastConversionPatterns(::mlir::RewritePatternSet &patterns,
                                    ::mlir::MLIRContext *ctx);

void populateMatMulConversionPatterns(::mlir::RewritePatternSet &patterns,
                                      ::mlir::MLIRContext *ctx);

void populateExpandConversionPatterns(::mlir::RewritePatternSet &patterns,
                                      ::mlir::MLIRContext *ctx);

// Populates the pattern converting `onnx.Shape` into a `hipsr.compute` whose
// body reads the input extents. The compute's placeholder gets its shape
// region populated here, because a compute op's result shape follows its body
// and so has no recipe hipsr-populate-shape-region could dispatch on.
void populateShapeConversionPatterns(::mlir::RewritePatternSet &patterns,
                                     ::mlir::MLIRContext *ctx);

// Populates the pattern converting `onnx.Reshape` into a `hipsr.compute` whose
// body collapses or expands the input's extents. Like the `onnx.Shape`
// pattern, it populates its placeholder's shape region itself.
void populateReshapeConversionPatterns(::mlir::RewritePatternSet &patterns,
                                       ::mlir::MLIRContext *ctx);

// Populates the pattern converting `onnx.Unsqueeze` into a `hipsr.compute`
// whose body splices unit axes into the input's extents. Like the `onnx.Shape`
// pattern, it populates its placeholder's shape region itself.
void populateUnsqueezeConversionPatterns(::mlir::RewritePatternSet &patterns,
                                         ::mlir::MLIRContext *ctx);

// Populates the pattern converting `onnx.Equal` into `hipsr.equal`.
void populateEqualConversionPatterns(::mlir::RewritePatternSet &patterns,
                                     ::mlir::MLIRContext *ctx);

// Populates the pattern converting `onnx.Transpose` into `hipsr.transpose`,
// materializing ONNX's default reverse permutation when `perm` is absent.
void populateTransposeConversionPatterns(::mlir::RewritePatternSet &patterns,
                                         ::mlir::MLIRContext *ctx);

// Populates the pattern converting `onnx.Gather` into `hipsr.gather`,
// normalizing a negative `axis` against the data rank.
void populateGatherConversionPatterns(::mlir::RewritePatternSet &patterns,
                                      ::mlir::MLIRContext *ctx);

// Populates the pattern converting `onnx.Slice` into `hipsr.slice`, resolving
// the window operands against the data extents. Covers constant windows over
// statically sized axes.
void populateSliceConversionPatterns(::mlir::RewritePatternSet &patterns,
                                     ::mlir::MLIRContext *ctx);

// Populates the pattern converting `onnx.ScatterND` into `hipsr.scatter_nd`.
// Covers the overwriting mode; the four reducing ones are rejected.
void populateScatterNDConversionPatterns(::mlir::RewritePatternSet &patterns,
                                         ::mlir::MLIRContext *ctx);

// Populates the pattern converting `onnx.NonZero` into `hipsr.nonzero`, which
// searches into a worst-case destination, followed by a `hipsr.compute` that
// narrows it to the positions found. The compute's placeholder is a barrier
// whose shape region is populated here, because that extent comes from a host
// read of the count `hipsr.nonzero` published.
void populateNonZeroConversionPatterns(::mlir::RewritePatternSet &patterns,
                                       ::mlir::MLIRContext *ctx);

void populateReturnConversionPatterns(::mlir::RewritePatternSet &patterns,
                                      ::mlir::MLIRContext *ctx);

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIPSR_ONNXTOHIPSR_H
