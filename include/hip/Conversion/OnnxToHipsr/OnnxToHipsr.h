/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_ONNXTOHIPSR_ONNXTOHIPSR_H
#define HIP_CONVERSION_ONNXTOHIPSR_ONNXTOHIPSR_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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

// These patterns take the converter but must not hand it to their
// OpConversionPattern base: a carried converter forces every operand through
// convertType, which overwrites the memory space its producer chose.
void populateOnnxToHipsrConstantPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns);

void populateCastConversionPatterns(const ::mlir::TypeConverter &typeConverter,
                                    ::mlir::RewritePatternSet &patterns,
                                    ::mlir::MLIRContext *ctx);

void populateMatMulConversionPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext *ctx);

void populateExpandConversionPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext *ctx);

void populateShapeConversionPatterns(const ::mlir::TypeConverter &typeConverter,
                                     ::mlir::RewritePatternSet &patterns,
                                     ::mlir::MLIRContext *ctx);

void populateReshapeConversionPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext *ctx);

void populateUnsqueezeConversionPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext *ctx);

void populateEqualConversionPatterns(const ::mlir::TypeConverter &typeConverter,
                                     ::mlir::RewritePatternSet &patterns,
                                     ::mlir::MLIRContext *ctx);

void populateTransposeConversionPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext *ctx);

void populateGatherConversionPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext *ctx);

void populateSliceConversionPatterns(const ::mlir::TypeConverter &typeConverter,
                                     ::mlir::RewritePatternSet &patterns,
                                     ::mlir::MLIRContext *ctx);

void populateScatterNDConversionPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext *ctx);

void populateReturnConversionPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext *ctx);

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIPSR_ONNXTOHIPSR_H
