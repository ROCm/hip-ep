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
/// Output rank is `q + r - indices_shape[-1] - 1 - batch_dims`, where
/// `q = rank(indices)` and `r = rank(data)`. The output layout is:
///
///   [ batch_dims | indices_outer_dims | data_tail ]
///
/// where `indices_outer_dims = indices.shape[batch_dims .. q-2]` and
/// `data_tail = data.shape[batch_dims + indices.shape[-1] .. r-1]`.
///
/// For dynamic result dims we source the size from whichever input dim is
/// authoritative for that position:
///
///   * batch region (i < batch_dims): take from data (or indices -- they
///     must agree by ONNX spec).
///   * indices-outer region (batch_dims <= i < q - 1): take from indices
///     at the same position `i`.
///   * data-tail region (i >= q - 1): take from data at
///     `i - (q - 1) + batch_dims + indices.shape[-1]`.
struct GatherNDToHip : public mlir::RewritePattern {
  GatherNDToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.GatherND", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value data = op->getOperand(0);
    mlir::Value indices = op->getOperand(1);
    auto indicesType =
        mlir::dyn_cast<mlir::RankedTensorType>(indices.getType());
    if (!indicesType || !indicesType.getElementType().isInteger(64)) {
      op->emitError("GatherND indices element type must be i64");
      return mlir::failure();
    }

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    int64_t batchDims = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("batch_dims"))
      batchDims = attr.getValue().getSExtValue();

    // The trailing index-tuple width must be statically known: it determines
    // the output rank, so a dynamic one leaves nothing to build a destination
    // from. The shared helper bails on the same condition.
    if (indicesType.isDynamicDim(indicesType.getRank() - 1))
      return rewriter.notifyMatchFailure(
          op, "GatherND requires static indices.shape[-1]");

    // Same helper that backs `GatherNDOp::reifyResultShapes`, so the
    // destination and the shape consumers observe cannot disagree.
    mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>> resultShape =
        mlir::hip::reifyGatherND(rewriter, loc, data, indices, batchDims);
    if (mlir::failed(resultShape))
      return rewriter.notifyMatchFailure(
          op, "GatherND batch_dims/indices layout is not reifiable");
    mlir::FailureOr<mlir::Value> init = createEmptyTensorFromReifiedShape(
        rewriter, loc, resultType, *resultShape);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, "GatherND result type is incompatible with the gathered shape");

    auto hipOp = mlir::hip::GatherNDOp::create(
        rewriter, loc, context, data, indices, *init,
        rewriter.getI64IntegerAttr(batchDims));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateGatherNDConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<GatherNDToHip>(ctx);
}

} // namespace hip
} // namespace mlir
