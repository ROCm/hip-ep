/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.slice.Tensor -> tensor.extract_slice
///
/// Signature: %out = "torch.aten.slice.Tensor"(%input, %dim, %start, %end, %step)
///   dim, start, end, step are torch.constant.int values.
struct TorchSliceToExtractSlice : public mlir::RewritePattern {
  TorchSliceToExtractSlice(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.slice.Tensor", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();

    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input is not a ranked tensor");

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Extract dim
    auto dimOpt = getTorchConstantInt(op->getOperand(1));
    if (!dimOpt)
      return rewriter.notifyMatchFailure(op, "dim must be a constant integer");
    int64_t dim = *dimOpt;
    int64_t rank = inputType.getRank();
    if (dim < 0)
      dim += rank;

    // Extract start
    auto startOpt = getTorchConstantInt(op->getOperand(2));
    if (!startOpt)
      return rewriter.notifyMatchFailure(op,
                                          "start must be a constant integer");
    int64_t start = *startOpt;
    if (start < 0)
      start += inputType.getDimSize(dim);

    // Extract end (may be INT64_MAX for "to the end")
    auto endOpt = getTorchConstantInt(op->getOperand(3));
    if (!endOpt)
      return rewriter.notifyMatchFailure(op, "end must be a constant integer");
    int64_t end = *endOpt;
    int64_t dimSize = inputType.getDimSize(dim);
    if (end > dimSize || end < 0)
      end = dimSize;

    // Step (optional, default 1)
    int64_t step = 1;
    if (op->getNumOperands() > 4) {
      auto stepOpt = getTorchConstantInt(op->getOperand(4));
      if (stepOpt)
        step = *stepOpt;
    }

    // Build offsets, sizes, strides for extract_slice
    llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
    for (int64_t i = 0; i < rank; ++i) {
      if (i == dim) {
        offsets.push_back(rewriter.getI64IntegerAttr(start));
        sizes.push_back(rewriter.getI64IntegerAttr(resultType.getDimSize(i)));
        strides.push_back(rewriter.getI64IntegerAttr(step));
      } else {
        offsets.push_back(rewriter.getI64IntegerAttr(0));
        sizes.push_back(rewriter.getI64IntegerAttr(inputType.getDimSize(i)));
        strides.push_back(rewriter.getI64IntegerAttr(1));
      }
    }

    auto extractOp = mlir::tensor::ExtractSliceOp::create(
        rewriter, loc, resultType, input, offsets, sizes, strides);
    rewriter.replaceOp(op, extractOp.getResult());
    return mlir::success();
  }
};

/// torch.aten.cat.default -> tensor.insert_slice chain
///
/// Signature: %out = "torch.aten.cat.default"(%tensors, %dim)
///   %tensors is a torch.prim.ListConstruct of tensors.
///   %dim is a torch.constant.int.
struct TorchCatToInsertSlice : public mlir::RewritePattern {
  TorchCatToInsertSlice(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.cat.default", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();

    // operand 0 = list of tensors (torch.prim.ListConstruct)
    mlir::Value listVal = op->getOperand(0);
    auto *listOp = listVal.getDefiningOp();
    if (!listOp ||
        listOp->getName().getStringRef() != "torch.prim.ListConstruct")
      return rewriter.notifyMatchFailure(
          op, "first operand must be torch.prim.ListConstruct");

    // operand 1 = dim
    auto dimOpt = getTorchConstantInt(op->getOperand(1));
    if (!dimOpt)
      return rewriter.notifyMatchFailure(op, "dim must be a constant integer");

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    int64_t rank = resultType.getRank();
    int64_t dim = *dimOpt;
    if (dim < 0)
      dim += rank;

    // Start with an empty output tensor
    mlir::Value output = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType());

    // Insert each input tensor at the right offset
    int64_t offset = 0;
    for (mlir::Value inputTensor : listOp->getOperands()) {
      auto inputType =
          mlir::cast<mlir::RankedTensorType>(inputTensor.getType());
      int64_t sliceSize = inputType.getDimSize(dim);

      llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
      for (int64_t i = 0; i < rank; ++i) {
        if (i == dim) {
          offsets.push_back(rewriter.getI64IntegerAttr(offset));
          sizes.push_back(rewriter.getI64IntegerAttr(sliceSize));
        } else {
          offsets.push_back(rewriter.getI64IntegerAttr(0));
          sizes.push_back(
              rewriter.getI64IntegerAttr(resultType.getDimSize(i)));
        }
        strides.push_back(rewriter.getI64IntegerAttr(1));
      }

      output = mlir::tensor::InsertSliceOp::create(
          rewriter, loc, inputTensor, output, offsets, sizes, strides);
      offset += sliceSize;
    }

    rewriter.replaceOp(op, output);
    return mlir::success();
  }
};

} // namespace

void populateTorchSliceCatConversionPatterns(mlir::RewritePatternSet &patterns,
                                              mlir::MLIRContext *ctx) {
  patterns.add<TorchSliceToExtractSlice, TorchCatToInsertSlice>(ctx);
}

} // namespace hip
} // namespace mlir
