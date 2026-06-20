/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Min -> hip.min (via MIOpen miopenOpTensor with miopenTensorOpMin)
///
/// Handles variadic inputs by pairwise chaining:
///   min(a, b, c) = min(min(a, b), c)
/// Single input is identity (pass through).
struct MinToHip : public mlir::RewritePattern {
  MinToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Min", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MinToHip::matchAndRewrite(mlir::Operation *op,
                          mlir::PatternRewriter &rewriter) const {
  unsigned numInputs = op->getNumOperands();
  if (numInputs == 0)
    return rewriter.notifyMatchFailure(op, "Min requires at least 1 input");

  // Single input: identity
  if (numInputs == 1) {
    rewriter.replaceOp(op, op->getOperand(0));
    return mlir::success();
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;
  mlir::Location loc = op->getLoc();

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Pairwise chaining: accumulate = min(accumulate, next_input)
  mlir::Value accumulate = op->getOperand(0);
  for (unsigned i = 1; i < numInputs; ++i) {
    mlir::Value rhs = op->getOperand(i);
    auto accType = mlir::cast<mlir::RankedTensorType>(accumulate.getType());
    mlir::RankedTensorType stepResultType =
        (i == numInputs - 1) ? resultType : accType;

    mlir::FailureOr<mlir::Value> initOrFailure = createBroadcastEmptyTensor(
        rewriter, loc, stepResultType, {accumulate, rhs});
    if (mlir::failed(initOrFailure))
      return rewriter.notifyMatchFailure(
          op, "Min: no ranked operand spans dynamic result dim");
    auto minOp = mlir::hip::MinOp::create(rewriter, loc, context, accumulate,
                                          rhs, *initOrFailure);
    accumulate = minOp->getResult(0);
  }

  rewriter.replaceOp(op, accumulate);
  return mlir::success();
}

} // namespace

void populateMinConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<MinToHip>(ctx);
}

} // namespace hip
} // namespace mlir
