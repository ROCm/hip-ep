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
    mlir::Value ctx = mul.getCtx();
    auto type = rankedType(mul->getResult(0));
    if (!type || !type.getElementType().isF16() || type.getRank() < 1 ||
        type.isDynamicDim(type.getRank() - 1))
      return rewriter.notifyMatchFailure(mul, "not f16");

    // Either operand may be the sigmoid, if it is a gate of the other.
    SigmoidOp sigmoid;
    mlir::Value x;
    for (auto [p, other] : {std::pair{mul.getLhs(), mul.getRhs()},
                            std::pair{mul.getRhs(), mul.getLhs()}}) {
      auto s = p.getDefiningOp<SigmoidOp>();
      if (!s || s.getCtx() != ctx || !hasSingleValueUse(p, mul) ||
          other == s.getX() || p.getType() != type ||
          other.getType() != type || s.getX().getType() != type ||
          !shapeQueriesRetargetable(p, s.getX()) ||
          !mlir::isMemoryEffectFree(s))
        continue;
      sigmoid = s;
      x = other;
      break;
    }
    if (!sigmoid)
      return rewriter.notifyMatchFailure(
          mul, "no same-shaped f16 sigmoid gate (other than SiLU)");
    mlir::Value gate = sigmoid.getX();
    mlir::Value prob = sigmoid->getResult(0);

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
