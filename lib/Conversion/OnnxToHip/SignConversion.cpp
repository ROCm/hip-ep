/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Sign -> hip.sign
/// Unary element-wise sign: Y = sign(X). Lowered through the unified unary
/// elementwise template (wrap_sign).
struct SignToHip : public mlir::RewritePattern {
  SignToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Sign", /*benefit=*/1, ctx) {}

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
          op, "onnx.Sign lowering expects a ranked tensor input");

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp = mlir::hip::SignOp::create(rewriter, loc, resultType, context,
                                           input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateSignConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<SignToHip>(ctx);
}

} // namespace hip
} // namespace mlir
