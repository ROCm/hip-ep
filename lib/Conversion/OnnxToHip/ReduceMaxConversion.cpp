/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "ReduceConversionUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceMax -> hip.reduce_max. The ONNX importer can leave the result
/// unranked (e.g. Phi's pos_ids_reformat ReduceMax(position_ids) feeding
/// GreaterOrEqual); it is inferred then (see inferReduceResultType).
struct ReduceMaxToHip : public mlir::RewritePattern {
  ReduceMaxToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceMax", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return convertOnnxReduction<mlir::hip::ReduceMaxOp>(
        op, rewriter, "ReduceMax", /*separable=*/true);
  }
};

} // namespace

void populateReduceMaxConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<ReduceMaxToHip>(ctx);
}

} // namespace hip
} // namespace mlir
