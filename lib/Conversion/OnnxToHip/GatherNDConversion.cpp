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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GatherNDToHip)
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
    auto indicesType = mlir::cast<mlir::RankedTensorType>(indices.getType());

    int64_t batchDims = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("batch_dims"))
      batchDims = attr.getValue().getSExtValue();

    int64_t q = indicesType.getRank();
    int64_t r = dataType.getRank();
    int64_t outRank = resultType.getRank();

    // The last dim of `indices` (call it `k`) controls how many leading data
    // dims each index tuple consumes. It must be statically known so we can
    // compute the data-tail mapping; dynamic-k GatherND is not expressible
    // with a single `tensor.empty` dynsize list.
    int64_t k = indicesType.getDimSize(q - 1);
    if (k == mlir::ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          op, "GatherND requires static indices.shape[-1]");

    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i = 0; i < outRank; ++i) {
      if (!resultType.isDynamicDim(i))
        continue;

      mlir::Value dim;
      if (i < batchDims) {
        // Batch dim -- ONNX requires data and indices to agree; prefer data
        // because we already have it bound for the tail case below.
        dim = mlir::tensor::DimOp::create(rewriter, loc, data, i);
      } else if (i < q - 1) {
        // Indices-outer dim: result[i] == indices[i] for batch_dims <= i < q-1.
        dim = mlir::tensor::DimOp::create(rewriter, loc, indices, i);
      } else {
        // Data-tail dim: result[i] == data[i - (q-1) + batch_dims + k].
        int64_t dataIdx = i - (q - 1) + batchDims + k;
        if (dataIdx < 0 || dataIdx >= r)
          return rewriter.notifyMatchFailure(
              op, "cannot resolve dynamic result dim from data tail");
        dim = mlir::tensor::DimOp::create(rewriter, loc, data, dataIdx);
      }
      dynSizes.push_back(dim);
    }
    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    auto hipOp = mlir::hip::GatherNDOp::create(
        rewriter, loc, context, data, indices, init,
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
