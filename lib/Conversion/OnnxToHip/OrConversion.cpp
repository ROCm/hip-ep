/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Or -> hip.or
///
/// Element-wise logical OR of two boolean tensors with NumPy-style
/// broadcasting. Dynamic output dimensions use the shared exact broadcast
/// selection, including when both aligned operand dimensions are dynamic.
struct OrToHip : public mlir::RewritePattern {
  OrToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Or", /*benefit=*/1, ctx) {}

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
        createOnnxBroadcastEmptyTensor(rewriter, loc, resultType, {a, b}, op);
    if (mlir::failed(initOrFailure))
      return rewriter.notifyMatchFailure(
          op, "Or: result type is incompatible with the broadcast shape");

    auto hipOp =
        mlir::hip::OrOp::create(rewriter, loc, context, a, b, *initOrFailure);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateOrConversionPatterns(RewritePatternSet &patterns,
                                  MLIRContext *ctx) {
  patterns.add<OrToHip>(ctx);
}

} // namespace hip
} // namespace mlir
