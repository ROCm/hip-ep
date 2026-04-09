/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.add.Tensor -> hip.add
struct TorchAddToHip : public mlir::RewritePattern {
  TorchAddToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.add.Tensor", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    // Check alpha == 1 (operand 2)
    auto alpha = getTorchConstantInt(op->getOperand(2));
    if (!alpha || *alpha != 1)
      return rewriter.notifyMatchFailure(
          op, "alpha must be constant 1 for direct add lowering");

    mlir::Location loc = op->getLoc();
    mlir::Value self = op->getOperand(0);
    mlir::Value other = op->getOperand(1);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    auto selfType = mlir::cast<mlir::RankedTensorType>(self.getType());
    mlir::Value source =
        (selfType.getRank() == resultType.getRank()) ? self : other;
    mlir::Value init =
        createEmptyTensorForTorch(rewriter, loc, resultType, source);

    auto hipOp = mlir::hip::AddOp::create(rewriter, loc, resultType, context,
                                          self, other, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

/// torch.aten.mul.Tensor -> hip.mul
struct TorchMulToHip : public mlir::RewritePattern {
  TorchMulToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.mul.Tensor", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value self = op->getOperand(0);
    mlir::Value other = op->getOperand(1);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    auto selfType = mlir::cast<mlir::RankedTensorType>(self.getType());
    mlir::Value source =
        (selfType.getRank() == resultType.getRank()) ? self : other;
    mlir::Value init =
        createEmptyTensorForTorch(rewriter, loc, resultType, source);

    auto hipOp = mlir::hip::MulOp::create(rewriter, loc, resultType, context,
                                          self, other, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

/// torch.aten.sub.Tensor -> hip.sub
struct TorchSubToHip : public mlir::RewritePattern {
  TorchSubToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.sub.Tensor", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    // Check alpha == 1 (operand 2)
    auto alpha = getTorchConstantInt(op->getOperand(2));
    if (!alpha || *alpha != 1)
      return rewriter.notifyMatchFailure(
          op, "alpha must be constant 1 for direct sub lowering");

    mlir::Location loc = op->getLoc();
    mlir::Value self = op->getOperand(0);
    mlir::Value other = op->getOperand(1);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    mlir::Value init =
        createEmptyTensorForTorch(rewriter, loc, resultType, self);

    auto hipOp = mlir::hip::SubOp::create(rewriter, loc, resultType, context,
                                          self, other, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateTorchElementwiseConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<TorchAddToHip, TorchMulToHip, TorchSubToHip>(ctx);
}

} // namespace hip
} // namespace mlir
