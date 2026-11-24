/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "ReduceConversionUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceMin -> hip.reduce_min
/// Reuses the same axes/keepdims/noop_with_empty_axes handling as ReduceMax.
struct ReduceMinToHip : public mlir::RewritePattern {
  ReduceMinToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceMin", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return convertOnnxReduction<mlir::hip::ReduceMinOp>(
        op, rewriter, "ReduceMin", /*separable=*/true);
  }
};

} // namespace

void populateReduceMinConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<ReduceMinToHip>(ctx);
}

} // namespace hip
} // namespace mlir
