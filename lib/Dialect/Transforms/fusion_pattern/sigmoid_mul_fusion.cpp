/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- sigmoid_mul_fusion.cpp - x * sigmoid(gate) into hip.sigmoid_mul ----===//
//
// Gated attention (Qwen3.5) scales its fp16 output by the sigmoid of a gate
// projection before the output projection:
//
//   Before:
//     %s = hip.sigmoid(%ctx) ins(%gate : tensor<1x?x4096xf16>) outs(...)
//     %y = hip.mul(%ctx) ins(%x, %s : ...) outs(%e : tensor<1x?x4096xf16>)
//   After:
//     %y = hip.sigmoid_mul(%ctx) ins(%x, %gate : ...) outs(%e : ...)
//
// hip.sigmoid_mul rounds the sigmoid to fp16 as hip.sigmoid does, so the
// result is unchanged. hip-hoist-dynamic-quant can then have it write the
// int8 activation of an int4 projection consuming %y.
//
// SiLU (x * sigmoid(x)) is left to the SwiGLU and gated-norm patterns, and a
// gate that would broadcast is not this op.
//
//===----------------------------------------------------------------------===//

#include "sigmoid_mul_fusion.hpp"

#include "fusion_utils.hpp"

namespace hip {
namespace fusion_transform {
namespace {

using namespace utils;
using mlir::hip::MulOp;
using mlir::hip::SigmoidMulOp;
using mlir::hip::SigmoidOp;

struct SigmoidMulFusion : public mlir::OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MulOp mul, mlir::PatternRewriter &rewriter) const override {
    SigmoidOp sigmoid;
    mlir::Value x;
    if (!matchEither(mul.getLhs(), mul.getRhs(), sigmoid, x))
      return rewriter.notifyMatchFailure(mul, "no hip.sigmoid operand");
    mlir::Value ctx = mul.getCtx();
    mlir::Value gate = sigmoid.getX();
    mlir::Value prob = sigmoid->getResult(0);
    if (sigmoid.getCtx() != ctx || !hasSingleValueUse(prob, mul))
      return rewriter.notifyMatchFailure(mul, "sigmoid has other uses");
    if (x == gate)
      return rewriter.notifyMatchFailure(mul, "SiLU");

    auto type = rankedType(mul->getResult(0));
    if (!type || !type.getElementType().isF16() || type.getRank() < 1 ||
        type.isDynamicDim(type.getRank() - 1) || x.getType() != type ||
        gate.getType() != type)
      return rewriter.notifyMatchFailure(mul, "not same-shaped f16");
    if (!shapeQueriesRetargetable(prob, gate))
      return rewriter.notifyMatchFailure(mul, "unexpected shape query");

    retargetShapeQueries(rewriter, prob, gate);
    rewriter.setInsertionPoint(mul);
    auto fused = SigmoidMulOp::create(rewriter, mul.getLoc(),
                                      mlir::TypeRange{type}, ctx, x, gate,
                                      mlir::ValueRange{mul.getOutput()},
                                      /*quant_block_size=*/nullptr);
    rewriter.replaceOp(mul, fused->getResults());
    mlir::Operation *chain[] = {sigmoid};
    eraseChain(rewriter, chain);
    return mlir::success();
  }
};

} // namespace

void populateSigmoidMulFusionPattern(mlir::RewritePatternSet &patterns,
                                     mlir::PatternBenefit benefit) {
  patterns.add<SigmoidMulFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
