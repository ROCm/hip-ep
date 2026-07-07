/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.GatherElements -> hip.gather_elements
struct GatherElementsToHip : public mlir::RewritePattern {
  GatherElementsToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.GatherElements", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
GatherElementsToHip::matchAndRewrite(mlir::Operation *op,
                                     mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);
  mlir::Value indices = op->getOperand(1);

  int64_t axis = 0;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = axisAttr.getSInt();

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto indicesType = mlir::cast<mlir::RankedTensorType>(indices.getType());

  llvm::SmallVector<mlir::Value> dynSizes;
  for (auto i : llvm::seq<int64_t>(0, resultType.getRank())) {
    if (resultType.isDynamicDim(i))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, indices, i));
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  auto gatherOp = mlir::hip::GatherElementsOp::create(
      rewriter, loc, context, data, indices, init,
      rewriter.getI64IntegerAttr(axis));

  rewriter.replaceOp(op, gatherOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateGatherElementsConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  patterns.add<GatherElementsToHip>(ctx);
}

} // namespace hip
} // namespace mlir
