/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Reciprocal -> hip.reciprocal
/// Converts ONNX reciprocal (y = 1/x, full signed IEEE domain) to HIP.
/// Runtime uses a HIP elementwise kernel (not MIOpen POWER activation).
struct ReciprocalToHip : public mlir::RewritePattern {
  ReciprocalToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Reciprocal", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// onnx.Sqrt -> hip.sqrt
/// ONNX element-wise sqrt; lowered to @wrap_power(0, 1, 0.5) and executed
/// with hip_elementwise_sqrt at runtime (not MIOpen POWER).
struct SqrtToHip : public mlir::RewritePattern {
  SqrtToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Sqrt", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReciprocalToHip::matchAndRewrite(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Reciprocal lowering expects a ranked tensor input");
  if (!mlir::isa<mlir::RankedTensorType>(op->getResult(0).getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Reciprocal lowering expects a ranked tensor result");
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::ReciprocalOp::create(rewriter, loc, resultType,
                                               context, input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

mlir::LogicalResult
SqrtToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Sqrt lowering expects a ranked tensor input");
  if (!mlir::isa<mlir::RankedTensorType>(op->getResult(0).getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Sqrt lowering expects a ranked tensor result");
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::SqrtOp::create(rewriter, loc, resultType, context,
                                         input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Neg -> hip.neg
/// Negation: y = -x. Lowered via wrap_power(alpha=0, beta=-1, gamma=1).
struct NegToHip : public mlir::RewritePattern {
  NegToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Neg", /*benefit=*/1, ctx) {}

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
          op, "onnx.Neg lowering expects a ranked tensor input");
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp = mlir::hip::NegOp::create(rewriter, loc, resultType, context,
                                          input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populatePowerConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<ReciprocalToHip, SqrtToHip, NegToHip>(ctx);
}

} // namespace hip
} // namespace mlir
