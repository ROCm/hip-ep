/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Reciprocal -> hip.reciprocal
/// Converts ONNX reciprocal operation (y = 1/x) to HIP power operation.
/// Runtime will use miopenActivationPOWER with gamma=-1.0.
struct ReciprocalToHip : public mlir::RewritePattern {
  ReciprocalToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Reciprocal", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// onnx.Sqrt -> hip.sqrt
/// Converts ONNX square root operation (y = √x) to HIP power operation.
/// Runtime will use miopenActivationPOWER with gamma=0.5.
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
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::SqrtOp::create(rewriter, loc, resultType, context,
                                         input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void mlir::hip::populatePowerConversionPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<ReciprocalToHip, SqrtToHip>(ctx);
}

} // namespace hip
} // namespace mlir
