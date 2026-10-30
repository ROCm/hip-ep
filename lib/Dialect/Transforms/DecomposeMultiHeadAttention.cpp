/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cmath>

namespace mlir::hip {
#define GEN_PASS_DEF_DECOMPOSEMULTIHEADATTENTIONPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

Value emptyTensor(PatternRewriter &rewriter, Location loc,
                  RankedTensorType type) {
  return tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                 type.getElementType());
}

Value reshapeByExpand(PatternRewriter &rewriter, Location loc, Value input,
                      RankedTensorType resultType,
                      ArrayRef<ReassociationIndices> reassociation) {
  return tensor::ExpandShapeOp::create(rewriter, loc, resultType, input,
                                       reassociation)
      .getResult();
}

Value reshapeByCollapse(PatternRewriter &rewriter, Location loc, Value input,
                        RankedTensorType resultType,
                        ArrayRef<ReassociationIndices> reassociation) {
  return tensor::CollapseShapeOp::create(rewriter, loc, resultType, input,
                                         reassociation)
      .getResult();
}

Value transposeTensor(PatternRewriter &rewriter, Location loc, Value ctx,
                      Value input, ArrayRef<int64_t> perm) {
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> shape;
  shape.reserve(perm.size());
  for (int64_t axis : perm)
    shape.push_back(inputType.getDimSize(axis));
  auto resultType = RankedTensorType::get(shape, inputType.getElementType());
  return TransposeOp::create(rewriter, loc, ctx, input,
                             emptyTensor(rewriter, loc, resultType),
                             rewriter.getI64ArrayAttr(perm))
      .getResult(0);
}

// Static separate query, key, and value only. Everything else keeps the
// runtime hip.multi_head_attention kernel.
class DecomposeMultiHeadAttention
    : public OpRewritePattern<MultiHeadAttentionOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MultiHeadAttentionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "present key, value, or qk");
    if (op.getBias() || op.getKeyPaddingMask() || op.getAttentionBias() ||
        op.getPastKey() || op.getPastValue() || op.getPastSequenceLength() ||
        op.getCacheIndirection() || op.getUnidirectional() != 0)
      return rewriter.notifyMatchFailure(
          op, "bias, mask, past, or causal attention");
    if (!op.getKey() || !op.getValue())
      return rewriter.notifyMatchFailure(op, "packed QKV");

    auto queryType = dyn_cast<RankedTensorType>(op.getQuery().getType());
    auto keyType = dyn_cast<RankedTensorType>(op.getKey().getType());
    auto valueType = dyn_cast<RankedTensorType>(op.getValue().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!queryType || !keyType || !valueType || !outputType ||
        !queryType.hasStaticShape() || !keyType.hasStaticShape() ||
        !valueType.hasStaticShape() || !outputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shape");
    if (queryType.getRank() != 3 || keyType.getRank() != 3 ||
        valueType.getRank() != 3 || outputType.getRank() != 3)
      return rewriter.notifyMatchFailure(op, "expected rank-3 Q, K, and V");

    Type elementType = queryType.getElementType();
    if (!isa<FloatType>(elementType) ||
        keyType.getElementType() != elementType ||
        valueType.getElementType() != elementType ||
        outputType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(op, "expected one float type");

    int64_t numHeads = op.getNumHeads();
    int64_t batch = queryType.getDimSize(0);
    int64_t seqQ = queryType.getDimSize(1);
    int64_t queryHidden = queryType.getDimSize(2);
    int64_t seqKv = keyType.getDimSize(1);
    if (numHeads <= 0 || queryHidden % numHeads != 0 ||
        keyType.getDimSize(0) != batch ||
        keyType.getDimSize(2) != queryHidden ||
        valueType.getDimSize(0) != batch || valueType.getDimSize(1) != seqKv)
      return rewriter.notifyMatchFailure(op, "Q, K, or V shape mismatch");

    int64_t headDim = queryHidden / numHeads;
    int64_t valueHidden = valueType.getDimSize(2);
    if (valueHidden % numHeads != 0)
      return rewriter.notifyMatchFailure(op, "value hidden is not heads * dim");
    int64_t headDimV = valueHidden / numHeads;
    if (outputType.getDimSize(0) != batch || outputType.getDimSize(1) != seqQ ||
        outputType.getDimSize(2) != valueHidden)
      return rewriter.notifyMatchFailure(op, "output shape mismatch");

    Location loc = op.getLoc();
    Value ctx = op.getCtx();
    SmallVector<ReassociationIndices> splitLast{{0}, {1}, {2, 3}};
    auto queryHeadsType =
        RankedTensorType::get({batch, seqQ, numHeads, headDim}, elementType);
    auto keyHeadsType =
        RankedTensorType::get({batch, seqKv, numHeads, headDim}, elementType);
    auto valueHeadsType =
        RankedTensorType::get({batch, seqKv, numHeads, headDimV}, elementType);

    Value queryHeads = reshapeByExpand(rewriter, loc, op.getQuery(),
                                       queryHeadsType, splitLast);
    Value keyHeads =
        reshapeByExpand(rewriter, loc, op.getKey(), keyHeadsType, splitLast);
    Value valueHeads = reshapeByExpand(rewriter, loc, op.getValue(),
                                       valueHeadsType, splitLast);
    // [B, N, Sq, D], [B, N, D, Sk], [B, N, Sk, Dv].
    Value queryBnsh =
        transposeTensor(rewriter, loc, ctx, queryHeads, {0, 2, 1, 3});
    Value keyT = transposeTensor(rewriter, loc, ctx, keyHeads, {0, 2, 3, 1});
    Value valueBnsh =
        transposeTensor(rewriter, loc, ctx, valueHeads, {0, 2, 1, 3});

    auto scoresType =
        RankedTensorType::get({batch, numHeads, seqQ, seqKv}, elementType);
    Value scores =
        MatmulOp::create(rewriter, loc, scoresType, ctx, queryBnsh, keyT,
                         emptyTensor(rewriter, loc, scoresType))
            .getResult(0);

    double scale = op.getScale().convertToFloat();
    if (scale == 0.0)
      scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    auto scaleType = RankedTensorType::get({}, elementType);
    APFloat scaleValue(scale);
    bool losesInfo = false;
    scaleValue.convert(cast<FloatType>(elementType).getFloatSemantics(),
                       APFloat::rmNearestTiesToEven, &losesInfo);
    Value scaleConst =
        ConstantOp::create(rewriter, loc, scaleType,
                           DenseElementsAttr::get(scaleType, scaleValue));
    scores = MulOp::create(rewriter, loc, scoresType, ctx, scores, scaleConst,
                           emptyTensor(rewriter, loc, scoresType))
                 .getResult(0);
    scores = MiopenSoftmaxOp::create(rewriter, loc, scoresType, ctx, scores,
                                     emptyTensor(rewriter, loc, scoresType))
                 .getResult(0);

    auto contextType =
        RankedTensorType::get({batch, numHeads, seqQ, headDimV}, elementType);
    Value context =
        MatmulOp::create(rewriter, loc, contextType, ctx, scores, valueBnsh,
                         emptyTensor(rewriter, loc, contextType))
            .getResult(0);
    Value outputHeads =
        transposeTensor(rewriter, loc, ctx, context, {0, 2, 1, 3});
    Value output =
        reshapeByCollapse(rewriter, loc, outputHeads, outputType, splitLast);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct DecomposeMultiHeadAttentionPass
    : public impl::DecomposeMultiHeadAttentionPassBase<
          DecomposeMultiHeadAttentionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeMultiHeadAttention>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::hip
