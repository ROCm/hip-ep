/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- AddRmsNormQ8Fusion.cpp - Residual/RMS/Q8 fusion -----------*- C++
//-*-===//

#include "GgmlToHipUtils.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace hip {
namespace {

/// Fuse the exact HIP tensor chain
///
///   residual   = hip.add(lhs, rhs)
///   normalized = hip.rms_norm(residual, scale)
///   q8          = hip.quantize_q8_1(normalized)
///
/// while retaining residual as the first result for its remaining users. The
/// normalized value must be used only by the Q8 preparation because the fused
/// kernel deliberately does not materialize it.
struct FuseAddRmsNormQ8 final : OpRewritePattern<QuantizeQ8_1Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(QuantizeQ8_1Op quantize,
                                PatternRewriter &rewriter) const override {
    if (quantize.getResultTensors().size() != 1)
      return rewriter.notifyMatchFailure(quantize,
                                         "requires one Q8_1 tensor result");
    Value normalized = quantize.getActivation();
    auto rmsNorm = normalized.getDefiningOp<RmsNormOp>();
    if (!rmsNorm || rmsNorm.getResultTensors().size() != 1 ||
        rmsNorm.getResultTensors().front() != normalized) {
      return rewriter.notifyMatchFailure(
          quantize, "requires a hip.rms_norm-produced activation");
    }

    SmallVector<tensor::DimOp> normalizedShapeUses;
    for (OpOperand &use : normalized.getUses()) {
      if (use.getOwner() == quantize.getOperation())
        continue;
      auto dim = dyn_cast<tensor::DimOp>(use.getOwner());
      if (!dim || dim.getSource() != normalized) {
        return rewriter.notifyMatchFailure(
            quantize,
            "requires normalized to have only Q8_1 and shape-only uses");
      }
      normalizedShapeUses.push_back(dim);
    }

    Value residual = rmsNorm.getInput();
    auto add = residual.getDefiningOp<AddOp>();
    if (!add || add.getResultTensors().size() != 1 ||
        add.getResultTensors().front() != residual) {
      return rewriter.notifyMatchFailure(
          quantize, "requires a hip.add-produced RMS input");
    }
    if (quantize.getCtx() != rmsNorm.getCtx() ||
        quantize.getCtx() != add.getCtx()) {
      return rewriter.notifyMatchFailure(quantize,
                                         "requires one shared HIP context");
    }

    auto function = quantize->getParentOfType<func::FuncOp>();
    if (!function)
      return rewriter.notifyMatchFailure(quantize,
                                         "requires a parent function");
    DominanceInfo dominance(function);
    SmallVector<tensor::DimOp> residualShapeUses;
    for (OpOperand &use : residual.getUses()) {
      if (use.getOwner() == rmsNorm.getOperation())
        continue;
      if (auto dim = dyn_cast<tensor::DimOp>(use.getOwner())) {
        if (dim.getSource() == residual) {
          residualShapeUses.push_back(dim);
          continue;
        }
      }
      if (!dominance.dominates(quantize.getOperation(), use.getOwner())) {
        return rewriter.notifyMatchFailure(
            quantize,
            "cannot move the residual definition past an earlier use");
      }
    }

    auto lhsType = dyn_cast<RankedTensorType>(add.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(add.getRhs().getType());
    auto residualType = dyn_cast<RankedTensorType>(residual.getType());
    auto normalizedType = dyn_cast<RankedTensorType>(normalized.getType());
    auto scaleType = dyn_cast<RankedTensorType>(rmsNorm.getScale().getType());
    auto q8Type = dyn_cast<RankedTensorType>(
        quantize.getResultTensors().front().getType());
    if (!lhsType || !rhsType || !residualType || !normalizedType ||
        !scaleType || !q8Type || lhsType.getRank() != 4 ||
        !lhsType.getElementType().isF32() || rhsType != lhsType ||
        residualType != lhsType || normalizedType != lhsType ||
        scaleType.getRank() != 1 || !scaleType.getElementType().isF32() ||
        !q8Type.getElementType().isUnsignedInteger(8) ||
        q8Type.getRank() != lhsType.getRank()) {
      return rewriter.notifyMatchFailure(
          quantize, "requires exact non-broadcasting f32 tensor shapes");
    }

    const int64_t k = quantize.getKAttr().getInt();
    const int64_t kBlockElements = kquant::kBlockElements;
    if (k <= 0 || k % kBlockElements != 0 || lhsType.getShape().back() != k ||
        scaleType.getDimSize(0) != k) {
      return rewriter.notifyMatchFailure(
          quantize, "requires a static K-compatible scale and residual");
    }
    int64_t q8RowBytes = 0;
    if (llvm::MulOverflow(k / kquant::kQ8_1BlockElements,
                          kquant::kQ8_1BlockBytes, q8RowBytes)) {
      return rewriter.notifyMatchFailure(
          quantize, "Q8_1 row byte cardinality overflows int64");
    }
    if (q8Type.getShape().back() != q8RowBytes) {
      return rewriter.notifyMatchFailure(quantize,
                                         "requires the exact Q8_1 row width");
    }
    for (int64_t dimension = 0; dimension < lhsType.getRank() - 1;
         ++dimension) {
      if (lhsType.getDimSize(dimension) != q8Type.getDimSize(dimension)) {
        return rewriter.notifyMatchFailure(
            quantize, "requires matching residual and Q8_1 prefix shapes");
      }
    }

    const int64_t axis = rmsNorm.getAxis();
    if ((axis != -1 && axis != lhsType.getRank() - 1) ||
        rmsNorm.getStashType() != 1) {
      return rewriter.notifyMatchFailure(
          quantize, "requires trailing-axis f32 RMS semantics");
    }
    const llvm::APFloat &epsilon = rmsNorm.getEpsilonAttr().getValue();
    if (!epsilon.isFinite() || epsilon.convertToDouble() < 0.0) {
      return rewriter.notifyMatchFailure(
          quantize, "requires finite non-negative epsilon");
    }

    if (add.getOutput().getType() != residualType ||
        rmsNorm.getOutput().getType() != normalizedType ||
        quantize.getOutput().getType() != q8Type) {
      return rewriter.notifyMatchFailure(
          quantize, "requires output init types to match their results");
    }

    auto fused = SkipRmsNormQ8_1Op::create(
        rewriter, quantize.getLoc(), TypeRange{residualType, q8Type},
        quantize.getCtx(), add.getLhs(), add.getRhs(), rmsNorm.getScale(),
        ValueRange{add.getOutput(), quantize.getOutput()}, quantize.getKAttr(),
        rmsNorm.getEpsilonAttr());
    for (NamedAttribute attribute : add->getAttrs()) {
      if (attribute.getName().getValue().starts_with("ggml."))
        fused->setAttr(attribute.getName(), attribute.getValue());
    }

    // Dynamic tensor.empty sizes are commonly derived through tensor.dim from
    // the soon-to-be-erased residual and normalized values. Their shapes are
    // proven identical to lhs above, so retarget these shape-only uses to the
    // dominating lhs without moving any data use across the fused operation.
    for (tensor::DimOp dim : residualShapeUses) {
      rewriter.modifyOpInPlace(
          dim, [&] { dim.getSourceMutable().assign(add.getLhs()); });
    }
    for (tensor::DimOp dim : normalizedShapeUses) {
      rewriter.modifyOpInPlace(
          dim, [&] { dim.getSourceMutable().assign(add.getLhs()); });
    }

    rewriter.replaceAllUsesWith(residual, fused.getResultTensors()[0]);
    rewriter.replaceOp(quantize, fused.getResultTensors()[1]);
    rewriter.eraseOp(rmsNorm);
    rewriter.eraseOp(add);
    return success();
  }
};

} // namespace

void populateHipAddRmsNormQ8FusionPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseAddRmsNormQ8>(patterns.getContext());
}

} // namespace hip
} // namespace mlir
