/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The gate and up K-quant projections of one Q8_1 activation feeding
// hip.swiglu, as hip.kquant_swiglu_q8_1.

#include "GgmlToHipUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace hip {
namespace {

static bool hasSupportedPackedWeight(Value value, int64_t n, int64_t k,
                                     StringRef format) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != 1 || !type.hasStaticShape() ||
      !type.getElementType().isUnsignedInteger(8))
    return false;
  std::optional<kquant::Format> packedFormat = kquant::parseFormat(format);
  if (packedFormat != kquant::Format::Q4K &&
      packedFormat != kquant::Format::Q6K)
    return false;
  int64_t expectedBytes = 0;
  return k > 0 && k % kquant::kBlockElements == 0 &&
         !llvm::MulOverflow(n, k / kquant::kBlockElements, expectedBytes) &&
         !llvm::MulOverflow(expectedBytes, kquant::getBlockBytes(*packedFormat),
                            expectedBytes) &&
         type.getNumElements() == expectedBytes;
}

static LogicalResult
collectShapeOnlyUses(Value value, Operation *consumer,
                     SmallVectorImpl<tensor::DimOp> &shapeUses,
                     unsigned consumerInputNumber, PatternRewriter &rewriter) {
  Operation *swiglu = consumer;
  for (OpOperand &use : value.getUses()) {
    if (use.getOwner() == consumer &&
        use.getOperandNumber() == consumerInputNumber)
      continue;
    auto dim = dyn_cast<tensor::DimOp>(use.getOwner());
    if (!dim || dim.getSource() != value ||
        !getConstantIntValue(dim.getIndex())) {
      return rewriter.notifyMatchFailure(
          swiglu, "projection results may have only constant-index "
                  "tensor.dim uses in addition to SwiGLU");
    }
    const int64_t dimension = *getConstantIntValue(dim.getIndex());
    if (dimension < 0 || dimension >= 4) {
      return rewriter.notifyMatchFailure(
          swiglu, "projection tensor.dim index is outside rank four");
    }
    shapeUses.push_back(dim);
  }
  return success();
}

static void retargetShapeUses(PatternRewriter &rewriter,
                              ArrayRef<tensor::DimOp> shapeUses,
                              Value quantizedActivation, int64_t n) {
  for (tensor::DimOp dim : shapeUses) {
    const int64_t dimension = *getConstantIntValue(dim.getIndex());
    if (dimension < 3) {
      rewriter.modifyOpInPlace(
          dim, [&] { dim.getSourceMutable().assign(quantizedActivation); });
      continue;
    }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(dim);
    Value constant = arith::ConstantIndexOp::create(rewriter, dim.getLoc(), n);
    rewriter.replaceOp(dim, constant);
  }
}

/// Fuse two packed projections feeding a plain F32 hip.swiglu. This runs
/// after GGML conversion, when shared Q8 preparation is explicit in SSA.
struct FuseKQuantSwiGlu final : OpRewritePattern<SwigluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SwigluOp swiglu,
                                PatternRewriter &rewriter) const override {
    if (swiglu.getResultTensors().size() != 1 || swiglu.isClamped() ||
        swiglu.getPromoteActivation())
      return rewriter.notifyMatchFailure(swiglu,
                                         "requires one plain SwiGLU result");

    auto gate = swiglu.getGate().getDefiningOp<KQuantMatmulQ8_1Op>();
    auto up = swiglu.getUp().getDefiningOp<KQuantMatmulQ8_1Op>();
    if (!gate || !up || gate == up || gate.getResultTensors().size() != 1 ||
        up.getResultTensors().size() != 1) {
      return rewriter.notifyMatchFailure(
          swiglu, "requires two distinct prepared-Q8 matmul producers");
    }
    if (gate.getCtx() != swiglu.getCtx() || up.getCtx() != swiglu.getCtx() ||
        gate.getQuantizedActivation() != up.getQuantizedActivation() ||
        gate.getSumMode() != up.getSumMode() ||
        gate.getRowsPerToken() != up.getRowsPerToken()) {
      return rewriter.notifyMatchFailure(
          swiglu, "requires one context, prepared-Q8 value, and sum mode");
    }
    if (gate.getK() != up.getK() || gate.getN() != up.getN()) {
      return rewriter.notifyMatchFailure(
          swiglu, "requires matching K and N projection dimensions");
    }

    Value gateResult = gate.getResultTensors().front();
    Value upResult = up.getResultTensors().front();
    auto q8Type =
        dyn_cast<RankedTensorType>(gate.getQuantizedActivation().getType());
    auto gateType = dyn_cast<RankedTensorType>(gateResult.getType());
    auto upType = dyn_cast<RankedTensorType>(upResult.getType());
    auto outputType =
        dyn_cast<RankedTensorType>(swiglu.getResultTensors().front().getType());
    if (!q8Type || !gateType || !upType || !outputType ||
        q8Type.getRank() != 4 || gateType.getRank() != 4 ||
        !q8Type.getElementType().isUnsignedInteger(8) ||
        !gateType.getElementType().isF32() || upType != gateType ||
        outputType != gateType) {
      return rewriter.notifyMatchFailure(
          swiglu, "requires matching rank-4 f32 projection and output shapes");
    }

    const int64_t k = gate.getK();
    const int64_t n = gate.getN();
    int64_t q8RowBytes = 0;
    if (k <= 0 || k % kquant::kBlockElements != 0 || n <= 0 ||
        llvm::MulOverflow(k / kquant::kQ8_1BlockElements,
                          kquant::kQ8_1BlockBytes, q8RowBytes) ||
        q8Type.getDimSize(3) != q8RowBytes || gateType.getDimSize(3) != n) {
      return rewriter.notifyMatchFailure(
          swiglu, "requires valid Q8_1 and projection dimensions");
    }
    for (int64_t dimension = 0; dimension < 3; ++dimension) {
      if (q8Type.getDimSize(dimension) != gateType.getDimSize(dimension)) {
        return rewriter.notifyMatchFailure(
            swiglu, "requires matching prepared-Q8 and output prefixes");
      }
    }
    if (!hasSupportedPackedWeight(gate.getPackedWeights(), n, k,
                                  gate.getFormat()) ||
        !hasSupportedPackedWeight(up.getPackedWeights(), n, k,
                                  up.getFormat())) {
      return rewriter.notifyMatchFailure(
          swiglu, "requires supported packed formats and exact byte counts");
    }
    if (gate.getOutput().getType() != gateType ||
        up.getOutput().getType() != upType ||
        swiglu.getOutput().getType() != outputType ||
        swiglu.getOutput() == gateResult || swiglu.getOutput() == upResult) {
      return rewriter.notifyMatchFailure(
          swiglu, "requires output init types to match their results");
    }

    SmallVector<tensor::DimOp> gateShapeUses;
    SmallVector<tensor::DimOp> upShapeUses;
    if (failed(collectShapeOnlyUses(gateResult, swiglu, gateShapeUses,
                                    /*swigluInputNumber=*/1, rewriter)) ||
        failed(collectShapeOnlyUses(upResult, swiglu, upShapeUses,
                                    /*swigluInputNumber=*/2, rewriter)))
      return failure();

    auto fused = KQuantSwigluQ8_1Op::create(
        rewriter, swiglu.getLoc(), swiglu.getCtx(),
        gate.getQuantizedActivation(), gate.getPackedWeights(),
        up.getPackedWeights(), swiglu.getOutput(), gate.getKAttr(),
        gate.getNAttr(), gate.getFormatAttr(), up.getFormatAttr(),
        gate.getSumModeAttr(), gate.getRowsPerTokenAttr());
    for (NamedAttribute attribute : swiglu->getAttrs()) {
      if (attribute.getName().getValue().starts_with("ggml."))
        fused->setAttr(attribute.getName(), attribute.getValue());
    }

    retargetShapeUses(rewriter, gateShapeUses, gate.getQuantizedActivation(),
                      n);
    retargetShapeUses(rewriter, upShapeUses, gate.getQuantizedActivation(), n);
    rewriter.replaceOp(swiglu, fused.getResultTensors());
    rewriter.eraseOp(gate);
    rewriter.eraseOp(up);
    return success();
  }
};

} // namespace

void populateHipKQuantSwiGluFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseKQuantSwiGlu>(patterns.getContext());
}

} // namespace hip
} // namespace mlir
