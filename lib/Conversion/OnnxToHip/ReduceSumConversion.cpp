/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "ReduceConversionUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceSum -> hip.reduce_sum. Constant axes that are not one
/// contiguous run become a chain of hip.reduce_sum, one per run; each partial
/// sum is rounded to the element type between runs.
struct ReduceSumToHip : public mlir::RewritePattern {
  ReduceSumToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceSum", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return convertOnnxReduction<mlir::hip::ReduceSumOp>(
        op, rewriter, "ReduceSum", /*separable=*/true);
  }
};

} // namespace

void populateReduceSumConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<ReduceSumToHip>(ctx);
}

} // namespace hip
} // namespace mlir
