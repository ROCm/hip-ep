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

  // Pick a source operand whose rank matches the result for tensor.empty
  // dynamic-dim extraction. Falls back through condition -> x -> y to handle
  // broadcasting from any operand having lower rank than the result.
  auto pickSource = [&](mlir::Value v) -> mlir::Value {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!t)
      return nullptr;
    return (t.getRank() == resultType.getRank()) ? v : nullptr;
  };
  mlir::Value source = pickSource(condition);
  if (!source)
    source = pickSource(x);
  if (!source)
    source = pickSource(y);
  if (!source)
    return rewriter.notifyMatchFailure(
        op, "onnx.Where lowering expects at least one operand whose rank "
            "matches the result");

  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, source);
  auto hipOp = mlir::hip::WhereOp::create(rewriter, loc, resultType, context,
                                          condition, x, y, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void mlir::hip::populateWhereConversionPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<WhereToHip>(ctx);
}

} // namespace hip
} // namespace mlir
