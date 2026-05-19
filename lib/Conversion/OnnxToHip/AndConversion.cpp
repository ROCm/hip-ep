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
/// broadcasting. Mirrors the EqualConversion pattern: pick whichever input
/// already matches the broadcast output rank as the shape source for
/// `createEmptyTensor`, so dynamic-shape models still get correct
/// `tensor.dim` extractions for the DPS init.
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

    // Pick the side whose rank already matches the broadcast output so
    // dynamic dims line up dim-by-dim for createEmptyTensor. When both
    // operands have the same rank as the result this is just `a`; when
    // one operand has a smaller rank (e.g. broadcasting a [N] mask against
    // a [B, N] tensor) we prefer the larger-rank operand.
    mlir::Value source = (aType.getRank() == resultType.getRank()) ? a : b;
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, source);

    auto hipOp = mlir::hip::AndOp::create(rewriter, loc, resultType, context, a,
                                          b, init);
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
