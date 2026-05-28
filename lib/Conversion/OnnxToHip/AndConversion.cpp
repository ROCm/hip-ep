/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.And -> hip.and
///
/// Element-wise logical AND of two boolean tensors with NumPy-style
/// broadcasting. Dynamic output dims are resolved via
/// `createBroadcastEmptyTensor` so each axis picks the non-broadcasting operand
/// (e.g. `[?x1] & [1x?] -> [?x?]`).
struct AndToHip : public mlir::RewritePattern {
  AndToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.And", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 2 inputs and 1 output");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value a = op->getOperand(0);
    mlir::Value b = op->getOperand(1);
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor output");

    auto aType = mlir::dyn_cast<mlir::RankedTensorType>(a.getType());
    auto bType = mlir::dyn_cast<mlir::RankedTensorType>(b.getType());
    if (!aType || !bType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor inputs");

    mlir::FailureOr<mlir::Value> initOrFailure =
        createBroadcastEmptyTensor(rewriter, loc, resultType, {a, b});
    if (mlir::failed(initOrFailure))
      return rewriter.notifyMatchFailure(
          op, "And: no ranked operand spans dynamic result dim");

    auto hipOp = mlir::hip::AndOp::create(rewriter, loc, resultType, context, a,
                                          b, *initOrFailure);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateAndConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<AndToHip>(ctx);
}

} // namespace hip
} // namespace mlir
