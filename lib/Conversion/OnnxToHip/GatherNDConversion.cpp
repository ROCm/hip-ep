/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.GatherND -> hip.gather_nd
///
/// Output rank is `q + r - indices_shape[-1] - 1 - batch_dims`. Because the
/// output rank generally differs from both input ranks, we cannot reuse
/// `createEmptyTensor` (which assumes a single rank-matching source). For
/// dynamic dims we right-align with `data`'s tail (after `batch_dims` and
/// the leading slice axes consumed by `indices`). When that mapping would
/// fall back into an indices-derived dim we conservatively bail out.
struct GatherNDToHip : public mlir::RewritePattern {
  GatherNDToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.GatherND", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value data = op->getOperand(0);
    mlir::Value indices = op->getOperand(1);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());

    int64_t batchDims = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("batch_dims"))
      batchDims = attr.getValue().getSExtValue();

    // Build dynSizes for the result. For each dynamic result dim, use the
    // matching data dim where the trailing alignment gives the same offset.
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      // Result dims: [batch_dims | gathered_q-1 | data_tail].
      // Tail is trailing of data; map result idx -> data idx by alignment.
      int64_t dataIdx = i + (dataType.getRank() - resultType.getRank());
      if (dataIdx < 0 || dataIdx >= dataType.getRank())
        return rewriter.notifyMatchFailure(
            op, "cannot resolve dynamic result dim from data tail");
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, data, dataIdx));
    }
    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        dynSizes);

    auto hipOp = mlir::hip::GatherNDOp::create(
        rewriter, loc, resultType, context, data, indices, init,
        rewriter.getI64IntegerAttr(batchDims));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void mlir::hip::populateGatherNDConversionPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx) {
  patterns.add<GatherNDToHip>(ctx);
}

} // namespace hip
} // namespace mlir
