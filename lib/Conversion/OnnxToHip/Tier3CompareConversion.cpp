/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Conversion patterns for the Tier 3/4 ops:
//
//   - onnx.Equal / Greater / Less / GreaterOrEqual / And -> hip.compare
//   - onnx.Where                                          -> hip.where
//   - onnx.LayerNormalization                             -> hip.layer_norm
//
// Compare/Where outputs are bool; ONNX represents them as i1 in MLIR but
// the runtime side expects 1 byte per element.  The runtime kernel writes
// one byte per element; downstream ops read it the same way.

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

// Kind values match hip_compare_kind_t.
static constexpr int64_t kCmpEq = 0;
static constexpr int64_t kCmpGt = 1;
static constexpr int64_t kCmpLt = 2;
static constexpr int64_t kCmpGe = 3;
static constexpr int64_t kCmpAnd = 4;

static LogicalResult buildCompare(Operation *op, PatternRewriter &rewriter,
                                  int64_t kind) {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  if (!isa<RankedTensorType>(lhs.getType()) ||
      !isa<RankedTensorType>(rhs.getType()))
    return rewriter.notifyMatchFailure(
        op, "hip.compare needs ranked tensor operands");
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  Value source = (lhsType.getRank() == resultType.getRank()) ? lhs : rhs;
  Value init = createEmptyTensor(rewriter, loc, resultType, source);
  auto hipOp = CompareOp::create(rewriter, loc, resultType, context, lhs, rhs,
                                  init, rewriter.getI64IntegerAttr(kind));
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

template <int64_t Kind>
struct OnnxCompareToHip : public RewritePattern {
  OnnxCompareToHip(MLIRContext *ctx, StringRef name)
      : RewritePattern(name, /*benefit=*/1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return buildCompare(op, rewriter, Kind);
  }
};

struct WhereToHip : public RewritePattern {
  WhereToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Where", /*benefit=*/1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3)
      return rewriter.notifyMatchFailure(op, "onnx.Where needs 3 operands");
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value cond = op->getOperand(0);
    Value x = op->getOperand(1);
    Value y = op->getOperand(2);
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "onnx.Where result not ranked");

    // Pick whichever input matches the result rank as the source for any
    // tensor.dim queries we need to materialise the empty tensor.
    Value source = x;
    if (auto t = dyn_cast<RankedTensorType>(x.getType())) {
      if (t.getRank() != resultType.getRank()) {
        if (auto t2 = dyn_cast<RankedTensorType>(y.getType());
            t2 && t2.getRank() == resultType.getRank())
          source = y;
      }
    }
    Location loc = op->getLoc();
    Value init = createEmptyTensor(rewriter, loc, resultType, source);
    auto hipOp = WhereOp::create(rewriter, loc, resultType, context, cond, x,
                                  y, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

struct LayerNormToHip : public RewritePattern {
  LayerNormToHip(MLIRContext *ctx)
      : RewritePattern("onnx.LayerNormalization", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    if (op->getNumOperands() < 2 || op->getNumOperands() > 3)
      return rewriter.notifyMatchFailure(
          op, "onnx.LayerNormalization expects 2 or 3 operands "
              "(input, gamma [, beta])");

    Value input = op->getOperand(0);
    Value gamma = op->getOperand(1);
    Value beta;
    if (op->getNumOperands() == 3) {
      Value third = op->getOperand(2);
      if (auto def = third.getDefiningOp())
        if (def->getName().getStringRef() == "onnx.NoValue")
          third = nullptr;
      beta = third;
    }

    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.LayerNormalization needs ranked tensors");

    int64_t axis = -1;
    if (auto a = op->getAttrOfType<IntegerAttr>("axis"))
      axis = a.getValue().getSExtValue();
    if (axis < 0)
      axis += inType.getRank();

    float epsilon = 1e-5f;
    if (auto e = op->getAttrOfType<FloatAttr>("epsilon"))
      epsilon = static_cast<float>(e.getValueAsDouble());

    Location loc = op->getLoc();
    Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp = LayerNormOp::create(rewriter, loc, resultType, context, input,
                                      gamma, beta, init,
                                      rewriter.getF32FloatAttr(epsilon),
                                      rewriter.getI64IntegerAttr(axis));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void mlir::hip::populateTier3CompareConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<OnnxCompareToHip<kCmpEq>>(ctx, "onnx.Equal");
  patterns.add<OnnxCompareToHip<kCmpGt>>(ctx, "onnx.Greater");
  patterns.add<OnnxCompareToHip<kCmpLt>>(ctx, "onnx.Less");
  patterns.add<OnnxCompareToHip<kCmpGe>>(ctx, "onnx.GreaterOrEqual");
  patterns.add<OnnxCompareToHip<kCmpAnd>>(ctx, "onnx.And");
  patterns.add<WhereToHip>(ctx);
  patterns.add<LayerNormToHip>(ctx);
}

} // namespace hip
} // namespace mlir
