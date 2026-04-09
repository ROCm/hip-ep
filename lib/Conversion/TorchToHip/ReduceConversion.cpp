/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.sum.dim_IntList -> hip.reduce_sum
struct TorchSumDimToHip : public mlir::RewritePattern {
  TorchSumDimToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.sum.dim_IntList", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    // Operands: (self, dim, keepdim, dtype)
    mlir::Value data = op->getOperand(0);

    // Extract dim list from torch.prim.ListConstruct
    auto dimListOpt = getTorchConstantIntList(op->getOperand(1));
    if (!dimListOpt)
      return rewriter.notifyMatchFailure(op,
                                         "dim must be a constant integer list");

    // Extract keepdim from torch.constant.bool
    auto keepdimOpt = getTorchConstantBool(op->getOperand(2));
    if (!keepdimOpt)
      return rewriter.notifyMatchFailure(op, "keepdim must be a constant bool");

    // Ignore dtype (operand 3) for now

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init =
        createEmptyTensorForTorch(rewriter, loc, resultType, data);

    // Create constant tensor for axes
    llvm::SmallVector<int64_t> &axesVec = *dimListOpt;
    auto axesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(axesVec.size())}, rewriter.getI64Type());
    auto axesAttr =
        mlir::DenseIntElementsAttr::get(axesType, llvm::ArrayRef(axesVec));
    mlir::Value axesOperand =
        mlir::arith::ConstantOp::create(rewriter, loc, axesType, axesAttr);

    // keepdims: true -> 1, false -> 0
    int64_t keepdims = *keepdimOpt ? 1 : 0;
    auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
    auto noopWithEmptyAxesAttr = rewriter.getI64IntegerAttr(0);

    auto hipOp = mlir::hip::ReduceSumOp::create(
        rewriter, loc, resultType, context, data, axesOperand, init,
        keepdimsAttr, noopWithEmptyAxesAttr);

    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateTorchReduceConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *ctx) {
  patterns.add<TorchSumDimToHip>(ctx);
}

} // namespace hip
} // namespace mlir
