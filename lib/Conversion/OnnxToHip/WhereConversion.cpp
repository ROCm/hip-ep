/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Where -> hip.where
/// ONNX Where: output[i] = condition[i] ? X[i] : Y[i].
/// Supports multidirectional (NumPy-style) broadcasting between condition,
/// X and Y. The condition tensor is bool (i1); X and Y share the result
/// element type.
struct WhereToHip : public mlir::RewritePattern {
  WhereToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Where", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
WhereToHip::matchAndRewrite(mlir::Operation *op,
                            mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "onnx.Where expects 3 operands and 1 result");

  mlir::Location loc = op->getLoc();
  mlir::Value condition = op->getOperand(0);
  mlir::Value x = op->getOperand(1);
  mlir::Value y = op->getOperand(2);

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return rewriter.notifyMatchFailure(
        op, "onnx.Where lowering expects a ranked tensor result");

  // Use the same NumPy broadcast rule as hip.where reification, including the
  // runtime case where an earlier dynamic operand resolves to one.
  mlir::FailureOr<mlir::Value> initOrFailure = createOnnxBroadcastEmptyTensor(
      rewriter, loc, resultType, {condition, x, y}, op);
  if (mlir::failed(initOrFailure))
    return rewriter.notifyMatchFailure(
        op, "onnx.Where: no ranked operand spans dynamic result dim");
  auto hipOp = mlir::hip::WhereOp::create(rewriter, loc, context, condition, x,
                                          y, *initOrFailure);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateWhereConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<WhereToHip>(ctx);
}

} // namespace hip
} // namespace mlir
