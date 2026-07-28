/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Max -> hip.max (via MIOpen miopenOpTensor with miopenTensorOpMax)
///
/// Handles variadic inputs by pairwise chaining:
///   max(a, b, c) = max(max(a, b), c)
/// Single input is identity (pass through).
///
/// Before:
///   %tmp = hip.max ... outs(%same_rank_as_a)
/// After:
///   %tmp_shape = broadcast_shape(%a, %b)
///   %tmp = hip.max ... outs(%empty_for_tmp_shape)
struct MaxToHip : public mlir::RewritePattern {
  MaxToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Max", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MaxToHip::matchAndRewrite(mlir::Operation *op,
                          mlir::PatternRewriter &rewriter) const {
  unsigned numInputs = op->getNumOperands();
  if (numInputs == 0)
    return rewriter.notifyMatchFailure(op, "Max requires at least 1 input");

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

  // Pairwise chaining: accumulate = max(accumulate, next_input)
  mlir::Value accumulate = op->getOperand(0);
  for (unsigned i = 1; i < numInputs; ++i) {
    mlir::Value rhs = op->getOperand(i);
    mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>> stepShape =
        mlir::hip::reifyBroadcastResultShape(rewriter, loc, {accumulate, rhs},
                                             [&]() { return op->emitError(); });
    if (mlir::failed(stepShape))
      return mlir::failure();

    bool isFinal = i == numInputs - 1;
    mlir::RankedTensorType stepResultType =
        isFinal ? resultType
                : getTensorTypeFromReifiedShape(*stepShape,
                                                resultType.getElementType());
    mlir::FailureOr<mlir::Value> init = createEmptyTensorFromReifiedShape(
        rewriter, loc, stepResultType, *stepShape);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, "Max result type is incompatible with broadcast shape");

    auto maxOp = mlir::hip::MaxOp::create(rewriter, loc, context, accumulate,
                                          rhs, *init);
    accumulate = maxOp->getResult(0);
  }

  rewriter.replaceOp(op, accumulate);
  return mlir::success();
}

} // namespace

void populateMaxConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<MaxToHip>(ctx);
}

} // namespace hip
} // namespace mlir
