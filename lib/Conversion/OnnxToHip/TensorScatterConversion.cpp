/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.TensorScatter -> hip.tensor_scatter
///
/// The output has the shape of `past_cache`; dynamic dims are taken from it.
/// `axis` (default -2) is normalized to [1, rank) here, ONNX forbids the
/// batch axis, and `mode` becomes the `circular` flag.
struct TensorScatterToHip : public RewritePattern {
  TensorScatterToHip(MLIRContext *ctx)
      : RewritePattern("onnx.TensorScatter", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumOperands() > 3 ||
        op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "expected (past_cache, update[, write_indices]), 1 output");
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();

    Value past = op->getOperand(0);
    Value update = op->getOperand(1);
    auto pastType = dyn_cast<RankedTensorType>(past.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!pastType || !resultType || !isa<RankedTensorType>(update.getType()))
      return rewriter.notifyMatchFailure(op, "expected ranked tensors");
    const int64_t rank = pastType.getRank();
    if (resultType.getRank() != rank)
      return rewriter.notifyMatchFailure(op, "result rank != past rank");

    int64_t axis = -2;
    if (auto attr = op->getAttrOfType<IntegerAttr>("axis"))
      axis = attr.getValue().getSExtValue();
    if (axis < 0)
      axis += rank;
    if (axis < 1 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis must be a non-batch dim");

    StringRef mode = "linear";
    if (auto attr = op->getAttrOfType<StringAttr>("mode"))
      mode = attr.getValue();
    if (mode != "linear" && mode != "circular")
      return rewriter.notifyMatchFailure(op, "unknown mode");

    Value writeIndices;
    if (op->getNumOperands() == 3 &&
        !isa<NoneType>(op->getOperand(2).getType()))
      writeIndices = op->getOperand(2);

    Location loc = op->getLoc();
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < rank; ++i)
      if (resultType.isDynamicDim(i))
        dynSizes.push_back(tensor::DimOp::create(rewriter, loc, past, i));
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType(), dynSizes);

    auto hipOp = TensorScatterOp::create(
        rewriter, loc, *ctxOrFailure, past, update, writeIndices, init,
        rewriter.getI64IntegerAttr(axis),
        rewriter.getBoolAttr(mode == "circular"));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void populateTensorScatterConversionPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx) {
  patterns.add<TensorScatterToHip>(ctx);
}

} // namespace hip
} // namespace mlir
