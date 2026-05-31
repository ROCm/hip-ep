/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Expand -> hip.expand
///
/// We trust the result type produced by ONNX shape inference. For dynamic
/// dims in the result we extract the corresponding entry from the `shape`
/// input tensor (right-aligned with the result rank, NumPy-style); leading
/// dims that are absent from `shape` fall back to the matching input dim.
struct ExpandToHip : public mlir::RewritePattern {
  ExpandToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Expand", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    mlir::Value shape = op->getOperand(1);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());

    int64_t resultRank = resultType.getRank();
    int64_t inputRank = inputType.getRank();

    // shape is a 1-D int tensor; for dynamic result dims we extract entries
    // from it (right-aligned with the result rank). When `shape` itself has
    // a dynamic length we cannot reason about it here, so bail out.
    auto shapeType = mlir::cast<mlir::RankedTensorType>(shape.getType());
    if (shapeType.getRank() != 1 || shapeType.isDynamicDim(0))
      return rewriter.notifyMatchFailure(
          op, "expand shape input must have static rank-1 type");
    int64_t shapeLen = shapeType.getDimSize(0);

    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i = 0; i < resultRank; ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      int64_t shapeIdx = i - (resultRank - shapeLen);
      mlir::Value dim;
      if (shapeIdx >= 0) {
        mlir::Value idx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, shapeIdx);
        mlir::Value extracted = mlir::tensor::ExtractOp::create(
            rewriter, loc, shape, mlir::ValueRange{idx});
        dim = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), extracted);
      } else {
        int64_t inputIdx = i - (resultRank - inputRank);
        if (inputIdx < 0)
          return rewriter.notifyMatchFailure(
              op, "cannot resolve dynamic dim from input or shape");
        dim = mlir::tensor::DimOp::create(rewriter, loc, input, inputIdx);
      }
      dynSizes.push_back(dim);
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    auto hipOp =
        mlir::hip::ExpandOp::create(rewriter, loc, context, input, shape, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateExpandConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<ExpandToHip>(ctx);
}

} // namespace hip
} // namespace mlir
