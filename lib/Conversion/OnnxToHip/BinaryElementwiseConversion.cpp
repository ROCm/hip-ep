/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers ONNX binary element-wise ops to a single hip.binary_elementwise op
// with a numeric `kind` discriminator and broadcasting handled at the
// runtime layer (kernel reads per-axis lhs/rhs strides; stride==0 means
// broadcast).
//
// Supported ONNX ops (Tier 1 in docs/kokoro_tts_plan.md):
//   onnx.Div, onnx.Pow
//
// Kind values must match hip_binary_kind_t in hip_custom_kernels.h:
//   0 = Div, 1 = Pow

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

static constexpr int64_t kKindDiv = 0;
static constexpr int64_t kKindPow = 1;

static LogicalResult buildBinary(Operation *op, PatternRewriter &rewriter,
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
        op, "hip.binary_elementwise needs ranked tensor operands");
  if (!isa<RankedTensorType>(op->getResult(0).getType()))
    return rewriter.notifyMatchFailure(
        op, "hip.binary_elementwise needs a ranked tensor result");

  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  // hip.binary_elementwise LLVM lowering requires static shapes; bail
  // on dynamic-result-type so dropUnsupportedOnnxOps replaces the op
  // with a tensor.empty placeholder rather than blowing up at HipToLLVM.
  if (!resultType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "hip.binary_elementwise lowering only supports static shapes");
  // Use broadcast-aware empty-tensor builder: walks lhs/rhs, picks the
  // first one that has a dim covering each dynamic output dim, falls
  // back to a 1-element constant when neither side has it (scalar bcast).
  Value init = createBroadcastEmptyTensor(rewriter, loc, resultType,
                                          {lhs, rhs});

  auto hipOp = BinaryElementwiseOp::create(
      rewriter, loc, resultType, context, lhs, rhs, init,
      rewriter.getI64IntegerAttr(kind));
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

template <int64_t Kind>
struct SimpleBinaryToHip : public RewritePattern {
  SimpleBinaryToHip(MLIRContext *ctx, StringRef name)
      : RewritePattern(name, /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return buildBinary(op, rewriter, Kind);
  }
};

} // namespace

void mlir::hip::populateBinaryElementwiseConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<SimpleBinaryToHip<kKindDiv>>(ctx, "onnx.Div");
  patterns.add<SimpleBinaryToHip<kKindPow>>(ctx, "onnx.Pow");
}

} // namespace hip
} // namespace mlir
