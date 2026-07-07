/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.OneHot -> hip.one_hot
struct OneHotToHip : public mlir::RewritePattern {
  OneHotToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.OneHot", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
OneHotToHip::matchAndRewrite(mlir::Operation *op,
                             mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value indices = op->getOperand(0);
  mlir::Value depth = op->getOperand(1);
  mlir::Value values = op->getOperand(2);

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto indicesType = mlir::cast<mlir::RankedTensorType>(indices.getType());

  int64_t axis = -1;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = axisAttr.getSInt();

  int64_t outRank = resultType.getRank();
  int64_t normAxis = axis;
  if (normAxis < 0)
    normAxis += outRank;

  llvm::SmallVector<mlir::Value> dynSizes;
  int64_t inDim = 0;
  for (int64_t outDim : llvm::seq<int64_t>(0, outRank)) {
    if (!resultType.isDynamicDim(outDim))
      continue;
    if (outDim == normAxis) {
      auto depthType = mlir::cast<mlir::RankedTensorType>(depth.getType());
      if (depthType.getRank() == 0)
        dynSizes.push_back(
            mlir::arith::ConstantIndexOp::create(rewriter, loc, 1));
      else
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, depth, 0));
    } else {
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, indices, inDim++));
    }
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  auto oneHotOp = mlir::hip::OneHotOp::create(rewriter, loc, context, indices,
                                              depth, values, init,
                                              rewriter.getI64IntegerAttr(axis));

  rewriter.replaceOp(op, oneHotOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateOneHotConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<OneHotToHip>(ctx);
}

} // namespace hip
} // namespace mlir
