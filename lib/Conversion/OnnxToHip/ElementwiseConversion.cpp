/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Add -> hip.miopen.add
struct AddToHip : public mlir::RewritePattern {
  AddToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Add", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// onnx.Mul -> hip.mul
struct MulToHip : public mlir::RewritePattern {
  MulToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Mul", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// onnx.Sub -> hip.sub
struct SubToHip : public mlir::RewritePattern {
  SubToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Sub", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
AddToHip::matchAndRewrite(mlir::Operation *op,
                          mlir::PatternRewriter &rewriter) const {
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
        op, "Add: no ranked operand spans dynamic result dim");

  return replaceWithBinaryElementwiseHipOp<mlir::hip::AddOp>(
      rewriter, op, context, a, b, *initOrFailure, resultType);
}

mlir::LogicalResult
MulToHip::matchAndRewrite(mlir::Operation *op,
                          mlir::PatternRewriter &rewriter) const {
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
        op, "Mul: no ranked operand spans dynamic result dim");

  return replaceWithBinaryElementwiseHipOp<mlir::hip::MulOp>(
      rewriter, op, context, a, b, *initOrFailure, resultType);
}

mlir::LogicalResult
SubToHip::matchAndRewrite(mlir::Operation *op,
                          mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value lhs = op->getOperand(0);
  mlir::Value rhs = op->getOperand(1);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::FailureOr<mlir::Value> initOrFailure =
      createBroadcastEmptyTensor(rewriter, loc, resultType, {lhs, rhs});
  if (mlir::failed(initOrFailure))
    return rewriter.notifyMatchFailure(
        op, "Sub: no ranked operand spans dynamic result dim");
  return replaceWithBinaryElementwiseHipOp<mlir::hip::SubOp>(
      rewriter, op, context, lhs, rhs, *initOrFailure, resultType);
}

} // namespace

void populateElementwiseConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<AddToHip, MulToHip, SubToHip>(ctx);
}

} // namespace hip
} // namespace mlir
