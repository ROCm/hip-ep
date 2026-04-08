/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.transpose.int -> hip.transpose
/// Operands: (self, dim0, dim1). Swaps the two specified dimensions.
struct TorchTransposeToHip : public mlir::RewritePattern {
  TorchTransposeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.transpose.int", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    // Operands: (self, dim0, dim1)
    mlir::Value data = op->getOperand(0);

    auto dim0Opt = getTorchConstantInt(op->getOperand(1));
    auto dim1Opt = getTorchConstantInt(op->getOperand(2));
    if (!dim0Opt || !dim1Opt)
      return rewriter.notifyMatchFailure(
          op, "dim0 and dim1 must be constant integers");

    int64_t dim0 = *dim0Opt;
    int64_t dim1 = *dim1Opt;

    // Normalize negative dimensions
    auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
    int64_t rank = dataType.getRank();
    if (dim0 < 0)
      dim0 += rank;
    if (dim1 < 0)
      dim1 += rank;

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Build the effective permutation to determine dynamic sizes
    // The permutation swaps dim0 and dim1
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t outDimIdx : llvm::seq<int64_t>(resultType.getRank())) {
      if (resultType.isDynamicDim(outDimIdx)) {
        int64_t srcDim = outDimIdx;
        if (outDimIdx == dim0)
          srcDim = dim1;
        else if (outDimIdx == dim1)
          srcDim = dim0;
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, data, srcDim));
      }
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    mlir::Value d0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, dim0);
    mlir::Value d1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, dim1);

    auto hipOp = mlir::hip::TransposeOp::create(rewriter, loc, resultType,
                                                context, d0, d1, data, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateTorchTransposeConversionPatterns(mlir::RewritePatternSet &patterns,
                                              mlir::MLIRContext *ctx) {
  patterns.add<TorchTransposeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
