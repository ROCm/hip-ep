/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.sigmoid -> hip.sigmoid
struct TorchSigmoidToHip : public mlir::RewritePattern {
  TorchSigmoidToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.sigmoid", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init =
        createEmptyTensorForTorch(rewriter, loc, resultType, input);

    auto hipOp = mlir::hip::SigmoidOp::create(rewriter, loc, resultType,
                                              context, input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

/// torch.aten.silu -> hip.silu
struct TorchSiluToHip : public mlir::RewritePattern {
  TorchSiluToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.silu", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init =
        createEmptyTensorForTorch(rewriter, loc, resultType, input);

    auto hipOp = mlir::hip::SiluOp::create(rewriter, loc, resultType, context,
                                           input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

/// torch.aten.softmax.int -> hip.miopen.softmax
/// Only supports softmax along the last dimension.
struct TorchSoftmaxToHip : public mlir::RewritePattern {
  TorchSoftmaxToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.softmax.int", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    // operand 1 = dim, operand 2 = dtype (ignored)
    auto dimOpt = getTorchConstantInt(op->getOperand(1));
    if (!dimOpt)
      return rewriter.notifyMatchFailure(op, "dim must be a constant integer");

    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    int64_t dim = *dimOpt;
    int64_t rank = inputType.getRank();

    // Normalize negative dim
    if (dim < 0)
      dim += rank;

    // MIOpen softmax only supports the last dimension
    if (dim != rank - 1)
      return rewriter.notifyMatchFailure(
          op, "hip.miopen.softmax only supports softmax along the last dim");

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init =
        createEmptyTensorForTorch(rewriter, loc, resultType, input);

    auto hipOp = mlir::hip::MiopenSoftmaxOp::create(rewriter, loc, resultType,
                                                    context, input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateTorchActivationConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<TorchSigmoidToHip, TorchSiluToHip, TorchSoftmaxToHip>(ctx);
}

} // namespace hip
} // namespace mlir
