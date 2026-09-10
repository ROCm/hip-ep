/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.IsInf -> hip.isinf
/// Unary element-wise infinity test: bool output, float input.
struct IsInfToHip : public mlir::RewritePattern {
  IsInfToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.IsInf", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.IsInf expects 1 operand and 1 result");

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "ranked tensor input required");
    if (!mlir::isa<mlir::FloatType>(inputType.getElementType()))
      return rewriter.notifyMatchFailure(op, "float element type required");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    int64_t detectNegative = 1;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("detect_negative"))
      detectNegative = attr.getInt();

    int64_t detectPositive = 1;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("detect_positive"))
      detectPositive = attr.getInt();

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

    auto hipOp = mlir::hip::IsInfOp::create(
        rewriter, loc, resultType, context, input, init,
        rewriter.getI64IntegerAttr(detectNegative),
        rewriter.getI64IntegerAttr(detectPositive));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateIsInfConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<IsInfToHip>(ctx);
}

} // namespace hip
} // namespace mlir
