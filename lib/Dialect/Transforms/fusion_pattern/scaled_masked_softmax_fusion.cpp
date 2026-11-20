/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- scaled_masked_softmax_fusion.cpp - scale, id mask, softmax ---------===//
//
//   Before:
//     %s = hip.mul(%ctx) ins(%scores, %scale : t, tensor<f16>) outs(%e0) : t
//     %r = tensor.expand_shape %ids [[0, 1]] : tensor<?xi64> into tensor<?x1xi64>
//     %c = tensor.expand_shape %ids [[0, 1]] : tensor<?xi64> into tensor<1x?xi64>
//     %m = hip.equal(%ctx) ins(%r, %c) outs(%e1 : tensor<?x?xui8>)
//     %w = hip.where(%ctx) ins(%m, %s, %fill : ..., t, tensor<f16>) outs(%e2) : t
//     %p = hip.miopen.softmax(%ctx) ins(%w : t) outs(%e3 : t) : t
//   After:
//     %p = hip.scaled_masked_softmax(%ctx) ins(%scores, %ids, %ids : t,
//            tensor<?xi64>, tensor<?xi64>) outs(%e3 : t)
//            {scale = ..., fill = ...} : t
//
// Where broadcasts its operands to its output, and the output type equals the
// scores type here, so a mask axis is either the matching scores axis or 1.
// That cannot be proven for dynamic axes at compile time; the runtime checks
// each id vector's length against the scores axis it indexes and accepts 1 as
// a broadcast, which is exactly Where's rule.
//
// Either equal operand may carry the row ids: lhs[r][0] == rhs[0][c] and
// rhs[r][0] == lhs[0][c] are the same mask once the roles are assigned.
//
//===----------------------------------------------------------------------===//

#include "scaled_masked_softmax_fusion.hpp"

#include "fusion_utils.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include <optional>

namespace hip {
namespace fusion_transform {
namespace {

/// The 1-D integer vector a [?,1] (unitAxis = 1) or [1,?] (unitAxis = 0)
/// expand_shape was built from, or null.
mlir::Value idVectorBehind(mlir::Value expanded, int64_t unitAxis) {
  auto expand = expanded.getDefiningOp<mlir::tensor::ExpandShapeOp>();
  if (!expand)
    return {};
  auto srcType =
      mlir::dyn_cast<mlir::RankedTensorType>(expand.getSrc().getType());
  auto dstType = mlir::dyn_cast<mlir::RankedTensorType>(expand.getType());
  if (!srcType || !dstType || srcType.getRank() != 1 ||
      dstType.getRank() != 2 || dstType.getDimSize(unitAxis) != 1)
    return {};
  if (!srcType.getElementType().isInteger(64))
    return {};
  return expand.getSrc();
}

std::optional<float> scalarConstant(mlir::Value value, mlir::Type elemType) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || type.getElementType() != elemType)
    return std::nullopt;
  std::optional<llvm::APFloat> v = getScalarFloatConstantValue(value);
  if (!v)
    return std::nullopt;
  bool lostInfo = false;
  v->convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven,
             &lostInfo);
  // f16 and f32 values are exact in f32; anything wider would be rounded.
  if (lostInfo)
    return std::nullopt;
  return v->convertToFloat();
}

struct ScaledMaskedSoftmaxFusion
    : public mlir::OpRewritePattern<mlir::hip::MiopenSoftmaxOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::MiopenSoftmaxOp softmax,
                  mlir::PatternRewriter &rewriter) const override {
    auto where = softmax.getInput().getDefiningOp<mlir::hip::WhereOp>();
    if (!where)
      return rewriter.notifyMatchFailure(softmax, "input is not hip.where");
    auto mul = where.getX().getDefiningOp<mlir::hip::MulOp>();
    if (!mul)
      return rewriter.notifyMatchFailure(softmax, "where x is not hip.mul");
    auto equal = where.getCondition().getDefiningOp<mlir::hip::EqualOp>();
    if (!equal)
      return rewriter.notifyMatchFailure(softmax,
                                         "where condition is not hip.equal");
    mlir::Value ctx = softmax.getCtx();
    if (!ctx || where.getCtx() != ctx || mul.getCtx() != ctx ||
        equal.getCtx() != ctx)
      return rewriter.notifyMatchFailure(softmax, "ops must share a context");
    if (!hasSingleValueUse(where->getResult(0), softmax) ||
        !hasSingleValueUse(mul->getResult(0), where))
      return rewriter.notifyMatchFailure(
          softmax, "scaled and masked scores must have one value use");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(softmax->getResult(0).getType());
    if (!resultType || resultType.getRank() < 2)
      return rewriter.notifyMatchFailure(softmax, "needs rank >= 2 scores");
    mlir::Type elemType = resultType.getElementType();
    if (!elemType.isF16() && !elemType.isF32())
      return rewriter.notifyMatchFailure(softmax, "only f16 and f32");
    if (where->getResult(0).getType() != resultType ||
        mul->getResult(0).getType() != resultType)
      return rewriter.notifyMatchFailure(
          softmax, "mul, where and softmax must share one type");

    mlir::Value scores;
    std::optional<float> scale;
    if ((scale = scalarConstant(mul.getRhs(), elemType)))
      scores = mul.getLhs();
    else if ((scale = scalarConstant(mul.getLhs(), elemType)))
      scores = mul.getRhs();
    else
      return rewriter.notifyMatchFailure(softmax,
                                         "mul has no scalar constant scale");
    if (scores.getType() != resultType)
      return rewriter.notifyMatchFailure(softmax, "scale must not broadcast");

    std::optional<float> fill = scalarConstant(where.getY(), elemType);
    if (!fill)
      return rewriter.notifyMatchFailure(softmax,
                                         "where y is not a scalar constant");

    auto condType =
        mlir::dyn_cast<mlir::RankedTensorType>(equal->getResult(0).getType());
    if (!condType || condType.getRank() != 2)
      return rewriter.notifyMatchFailure(softmax, "mask must be rank 2");
    mlir::Value rowIds = idVectorBehind(equal.getLhs(), /*unitAxis=*/1);
    mlir::Value colIds = idVectorBehind(equal.getRhs(), /*unitAxis=*/0);
    if (!rowIds || !colIds) {
      rowIds = idVectorBehind(equal.getRhs(), /*unitAxis=*/1);
      colIds = idVectorBehind(equal.getLhs(), /*unitAxis=*/0);
    }
    if (!rowIds || !colIds)
      return rewriter.notifyMatchFailure(
          softmax, "mask is not an outer comparison of two id vectors");

    retargetShapeQueries(rewriter, where->getResult(0), scores);
    retargetShapeQueries(rewriter, mul->getResult(0), scores);
    auto fused = mlir::hip::ScaledMaskedSoftmaxOp::create(
        rewriter, softmax.getLoc(), resultType, ctx, scores, rowIds, colIds,
        softmax.getOutput(), rewriter.getF32FloatAttr(*scale),
        rewriter.getF32FloatAttr(*fill));
    rewriter.replaceOp(softmax, fused.getResult(0));
    rewriter.eraseOp(where);
    rewriter.eraseOp(mul);
    // The mask and its expansions may also feed another block's where; the
    // greedy driver drops them once nothing reads them.
    return mlir::success();
  }
};

} // namespace

void populateScaledMaskedSoftmaxFusionPattern(
    mlir::RewritePatternSet &patterns, mlir::PatternBenefit benefit) {
  patterns.add<ScaledMaskedSoftmaxFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
