/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.index_select -> hip.gather
struct TorchIndexSelectToHip : public mlir::RewritePattern {
  TorchIndexSelectToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.index_select", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    // Operands: (self, dim, index)
    mlir::Value data = op->getOperand(0);
    mlir::Value indices = op->getOperand(2);

    // Get dim from torch.constant.int
    auto dimOpt = getTorchConstantInt(op->getOperand(1));
    if (!dimOpt)
      return rewriter.notifyMatchFailure(op, "dim must be a constant integer");
    int64_t axis = *dimOpt;

    auto axisAttr = rewriter.getI64IntegerAttr(axis);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
    auto indicesType = mlir::cast<mlir::RankedTensorType>(indices.getType());

    // Normalize negative axis for dimension calculations
    int64_t normalizedAxis = axis < 0 ? axis + dataType.getRank() : axis;

    // Create output tensor with dynamic shape support
    // Output shape: [data[0:axis], indices.shape, data[axis+1:]]
    llvm::SmallVector<mlir::Value> dynSizes;
    int64_t outDimIdx = 0;

    // Copy dimensions before axis from data
    for (auto i : llvm::seq<int64_t>(0, normalizedAxis)) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
      outDimIdx++;
    }
    // Copy all dimensions from indices
    for (auto i : llvm::seq<int64_t>(0, indicesType.getRank())) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, indices, i));
      outDimIdx++;
    }
    // Copy dimensions after axis from data
    for (auto i : llvm::seq<int64_t>(normalizedAxis + 1, dataType.getRank())) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
      outDimIdx++;
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    auto gatherOp = mlir::hip::GatherOp::create(
        rewriter, loc, resultType, context, data, indices, init, axisAttr);

    rewriter.replaceOp(op, gatherOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateTorchGatherConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *ctx) {
  patterns.add<TorchIndexSelectToHip>(ctx);
}

} // namespace hip
} // namespace mlir
