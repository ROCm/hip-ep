/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Less -> hip.less
struct LessToHip : public mlir::RewritePattern {
  LessToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Less", /*benefit=*/1, ctx) {}

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

    mlir::FailureOr<mlir::Value> initOrFailure =
        createBroadcastEmptyTensor(rewriter, loc, resultType, {a, b});
    if (mlir::failed(initOrFailure))
      return rewriter.notifyMatchFailure(
          op, "Less: no ranked operand spans dynamic result dim");

    return replaceWithBroadcastBinaryHipOp<mlir::hip::LessOp>(
        rewriter, op, "Less", context, a, b, *initOrFailure, resultType);
  }
};

} // namespace

void populateLessConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<LessToHip>(ctx);
}

} // namespace hip
} // namespace mlir
