/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Div -> hip.div
struct DivToHip : public mlir::RewritePattern {
  DivToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Div", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value a = op->getOperand(0);
    mlir::Value b = op->getOperand(1);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    auto aType = mlir::cast<mlir::RankedTensorType>(a.getType());
    mlir::Value source = (aType.getRank() == resultType.getRank()) ? a : b;
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, source);

    auto hipOp =
        mlir::hip::DivOp::create(rewriter, loc, context, a, b, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateDivConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<DivToHip>(ctx);
}

} // namespace hip
} // namespace mlir
