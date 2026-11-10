/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "ReduceConversionUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceMean -> hip.reduce_mean
///
/// Direct, dim-tolerant conversion: the division by the reduced-element count
/// happens inside the runtime kernel, so (unlike the retired
/// ReduceMeanToReduceSumDiv decomposition) no static reduce axis is required.
/// Result dims are refined later by --hip-infer-shapes through the upstream
/// MLIR ReifyRankedShapedTypeOpInterface / InferTypeOpInterface that
/// Hip_DpsOp_Reduction implements, so no ONNX-level shape refinement is needed.
///
/// `axes` is always materialized as an operand for the DPS op: opset >= 18
/// already supplies it as an operand; for older opsets (or the all-axes
/// default) it is built here from the `axes` attribute into an i64 constant.
/// Constant axes that are not one contiguous run become a chain of
/// hip.reduce_mean, one per run (see ReduceConversionUtils.h).
///
/// Before (opset < 18, axes as attribute):
///   %y = "onnx.ReduceMean"(%x) {axes = [1], keepdims = 1}
///          : (tensor<?x4096xf16>) -> tensor<?x1xf16>
/// After:
///   %axes = arith.constant dense<[1]> : tensor<1xi64>
///   %init = tensor.empty(%n) : tensor<?x1xf16>
///   %y    = hip.reduce_mean(%ctx) ins(%x, %axes) outs(%init) {keepdims = 1}
struct ReduceMeanToHip : public mlir::RewritePattern {
  ReduceMeanToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceMean", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return convertOnnxReduction<mlir::hip::ReduceMeanOp>(
        op, rewriter, "ReduceMean", /*separable=*/true);
  }
};

} // namespace

void populateReduceMeanConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<ReduceMeanToHip>(ctx);
}

} // namespace hip
} // namespace mlir
