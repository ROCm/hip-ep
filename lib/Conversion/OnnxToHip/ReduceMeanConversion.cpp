/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceMean -> hip.reduce_mean.
/// Mirrors the ReduceMax/ReduceSum lowering: optional `axes` may arrive as
/// an operand (opset 18+) or as an `axes` attribute (older opsets); when
/// absent we synthesize a constant tensor (full-axis reduce if
/// `noop_with_empty_axes == 0`, empty axes otherwise).
struct ReduceMeanToHip : public mlir::RewritePattern {
  ReduceMeanToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceMean", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReduceMeanToHip::matchAndRewrite(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, data);

  int64_t noopWithEmptyAxes = 0;
  if (auto noopAttr =
          op->getAttrOfType<mlir::IntegerAttr>("noop_with_empty_axes"))
    noopWithEmptyAxes = noopAttr.getSInt();

  mlir::Value axesOperand;
  if (op->getNumOperands() > 1) {
    axesOperand = op->getOperand(1);
  } else {
    llvm::SmallVector<int64_t> axesVec;
    if (auto axesAttr = op->getAttrOfType<mlir::ArrayAttr>("axes")) {
      for (auto a : axesAttr)
        axesVec.push_back(
            mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
    } else if (noopWithEmptyAxes == 0) {
      auto inputType = mlir::cast<mlir::RankedTensorType>(data.getType());
      for (int64_t i : llvm::seq<int64_t>(inputType.getRank()))
        axesVec.push_back(i);
    }
    auto axesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(axesVec.size())}, rewriter.getI64Type());
    auto axesAttr =
        mlir::DenseIntElementsAttr::get(axesType, llvm::ArrayRef(axesVec));
    axesOperand =
        mlir::arith::ConstantOp::create(rewriter, loc, axesType, axesAttr);
  }

  int64_t keepdims = 1;
  if (auto keepdimsAttr = op->getAttrOfType<mlir::IntegerAttr>("keepdims"))
    keepdims = keepdimsAttr.getSInt();

  auto hipOp = mlir::hip::ReduceMeanOp::create(
      rewriter, loc, resultType, context, data, axesOperand, init,
      rewriter.getI64IntegerAttr(keepdims),
      rewriter.getI64IntegerAttr(noopWithEmptyAxes));
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateReduceMeanConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<ReduceMeanToHip>(ctx);
}

} // namespace hip
} // namespace mlir
