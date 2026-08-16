/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ScatterElements -> hip.scatter_elements
struct ScatterElementsToHip : public mlir::RewritePattern {
  ScatterElementsToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ScatterElements", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ScatterElementsToHip::matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "expected 3 inputs (data, indices, updates), 1 output");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);
  mlir::Value indices = op->getOperand(1);
  mlir::Value updates = op->getOperand(2);

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  if (!resultType || !dataType)
    return rewriter.notifyMatchFailure(
        op, "expected ranked tensor types for data and output");
  if (resultType.getRank() != dataType.getRank())
    return rewriter.notifyMatchFailure(
        op, "ScatterElements requires result rank == data rank");

  mlir::FailureOr<mlir::Value> init =
      createSameShapeEmptyTensor(rewriter, loc, resultType, data);
  if (mlir::failed(init))
    return rewriter.notifyMatchFailure(
        op, "result shape contradicts ScatterElements data shape");

  int64_t axis = 0;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = axisAttr.getSInt();

  mlir::StringAttr reductionAttr;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("reduction"))
    reductionAttr = attr;
  else
    reductionAttr = rewriter.getStringAttr("none");

  auto scatterOp = mlir::hip::ScatterElementsOp::create(
      rewriter, loc, context, data, indices, updates, *init,
      rewriter.getI64IntegerAttr(axis), reductionAttr);
  rewriter.replaceOp(op, scatterOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateScatterElementsConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<ScatterElementsToHip>(ctx);
}

} // namespace hip
} // namespace mlir
