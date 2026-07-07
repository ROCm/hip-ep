/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Compress -> hip.compress
struct CompressToHip : public mlir::RewritePattern {
  CompressToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Compress", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
CompressToHip::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  mlir::Value condition = op->getOperand(1);

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());

  bool flatten = !op->hasAttr("axis");
  int64_t axis = 0;
  if (!flatten) {
    axis = op->getAttrOfType<mlir::IntegerAttr>("axis").getSInt();
    if (axis < 0)
      axis += inputType.getRank();
  }

  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i : llvm::seq<int64_t>(0, resultType.getRank())) {
    if (!resultType.isDynamicDim(i))
      continue;
    if (flatten || i == axis)
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, condition, 0));
    else
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, input, i));
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  auto compressOp = mlir::hip::CompressOp::create(
      rewriter, loc, context, input, condition, init,
      rewriter.getI64IntegerAttr(axis),
      rewriter.getBoolAttr(flatten));

  rewriter.replaceOp(op, compressOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateCompressConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<CompressToHip>(ctx);
}

} // namespace hip
} // namespace mlir
