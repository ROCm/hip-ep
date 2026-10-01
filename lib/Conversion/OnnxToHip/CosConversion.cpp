/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Cos -> hip.cos
/// Unary element-wise cosine: Y = cos(X). Float types.
struct CosToHip : public mlir::RewritePattern {
  CosToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Cos", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
      return rewriter.notifyMatchFailure(
          op, "onnx.Cos lowering expects a ranked tensor input");
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto init = createSameShapeEmptyTensor(rewriter, loc, resultType, input);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, "Cos result type must match the input shape");
    auto hipOp = mlir::hip::CosOp::create(rewriter, loc, context, input, *init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateCosConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<CosToHip>(ctx);
}

} // namespace hip
} // namespace mlir
