/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- RmsNormScaleFusion.cpp - GGML RMS scale fusion ------------*- C++
//-*-===//

#include "GgmlToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

static bool haveEquivalentShape(Value lhs, Value rhs) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType || lhsType != rhsType)
    return false;

  for (int64_t dimension = 0; dimension < lhsType.getRank(); ++dimension) {
    if (!haveEquivalentGgmlDimensionExtent(lhs, dimension, rhs, dimension))
      return false;
  }
  return true;
}

/// A gamma of [1, 1, 1, H], or [1, 1, G, H] for a grouped norm whose rows
/// cycle through the G groups (Mamba-2's per-group output norm).
static bool isExactRmsGamma(Value gamma, int64_t groups, int64_t hiddenSize) {
  auto type = dyn_cast<RankedTensorType>(gamma.getType());
  return type && type.getRank() == 4 && type.getElementType().isF32() &&
         type.getDimSize(0) == 1 && type.getDimSize(1) == 1 &&
         (type.getDimSize(2) == 1 ||
          (!ShapedType::isDynamic(groups) && type.getDimSize(2) == groups)) &&
         type.getDimSize(3) == hiddenSize;
}

/// Fold the exact GGML representation
///
///   scaled = mul(rms_norm(input), gamma)
///
/// into hip.rms_norm(input, gamma). GGML RMS_NORM is unscaled, while the HIP
/// operation applies its scale in the normalization kernel. Restricting gamma
/// to [1, 1, 1, H] makes the GGML broadcast exactly the HIP per-feature scale.
struct FuseRmsNormScale final : OpRewritePattern<ggml::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ggml::MulOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsRms = op.getLhs().getDefiningOp<ggml::RmsNormOp>();
    auto rhsRms = op.getRhs().getDefiningOp<ggml::RmsNormOp>();
    if (static_cast<bool>(lhsRms) == static_cast<bool>(rhsRms)) {
      return rewriter.notifyMatchFailure(
          op, "requires exactly one RMS_NORM operand");
    }

    ggml::RmsNormOp rms = lhsRms ? lhsRms : rhsRms;
    Value gamma = lhsRms ? op.getRhs() : op.getLhs();
    if (!rms->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          op, "requires an exclusively consumed RMS_NORM result");
    }

    auto inputType = dyn_cast<RankedTensorType>(rms.getInput().getType());
    auto normalizedType = dyn_cast<RankedTensorType>(rms.getResult().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !normalizedType || !resultType ||
        inputType.getRank() != 4 || normalizedType.getRank() != 4 ||
        resultType.getRank() != 4 || !inputType.getElementType().isF32() ||
        !normalizedType.getElementType().isF32() ||
        !resultType.getElementType().isF32() || inputType != normalizedType ||
        normalizedType != resultType) {
      return rewriter.notifyMatchFailure(
          op, "requires matching rank-4 f32 input, normalized, and result "
              "tensors");
    }

    const int64_t hiddenSize = normalizedType.getDimSize(3);
    const int64_t groups = normalizedType.getDimSize(2);
    if (ShapedType::isDynamic(hiddenSize) || hiddenSize <= 0 ||
        !isExactRmsGamma(gamma, groups, hiddenSize)) {
      return rewriter.notifyMatchFailure(
          op, "requires exact rank-4 [1,1,1,H] or [1,1,G,H] f32 gamma");
    }
    const int64_t gammaRows =
        cast<RankedTensorType>(gamma.getType()).getDimSize(2);
    if (!haveEquivalentShape(rms.getInput(), rms.getResult()) ||
        !haveEquivalentShape(rms.getResult(), op.getResult())) {
      return rewriter.notifyMatchFailure(
          op, "requires shape-preserving GGML provenance");
    }

    FailureOr<Value> context = getHipContext(op, rewriter);
    if (failed(context))
      return failure();

    // A grouped gamma is hip.rms_norm's [G, H] scale, row r taking group
    // r % G.
    Value gammaVector =
        gammaRows == 1
            ? Value(tensor::CollapseShapeOp::create(
                  rewriter, op.getLoc(),
                  RankedTensorType::get({hiddenSize}, rewriter.getF32Type()),
                  gamma, SmallVector<ReassociationIndices>{{0, 1, 2, 3}}))
            : Value(tensor::CollapseShapeOp::create(
                  rewriter, op.getLoc(),
                  RankedTensorType::get({gammaRows, hiddenSize},
                                        rewriter.getF32Type()),
                  gamma, SmallVector<ReassociationIndices>{{0, 1, 2}, {3}}));
    Value init = createGgmlEmptyTensor(rewriter, op.getLoc(), resultType,
                                       rms.getInput());
    if (!init)
      return failure();

    auto fused = hip::RmsNormOp::create(
        rewriter, op.getLoc(), *context, rms.getInput(), gammaVector, init,
        rewriter.getI64IntegerAttr(-1), rms.getEpsilonAttr(),
        rewriter.getI64IntegerAttr(1));
    copyGgmlResultProvenanceAttrs(op, fused);

    rewriter.replaceOp(op, fused.getResultTensors());
    rewriter.eraseOp(rms);
    return success();
  }
};

} // namespace

void populateGgmlRmsNormScaleFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseRmsNormScale>(patterns.getContext());
}

} // namespace hip
} // namespace mlir
