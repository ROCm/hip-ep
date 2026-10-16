/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- gated_rms_norm_fusion.cpp - SiLU-gated RMS norm into one op --------===//
//
// A gated-delta-net layer gates its normalized attention output with the
// SiLU of a projection before the output projection. Exported per op, that
// is seven launches at decode; hip.gated_rms_norm is one, and it can also
// write the quantized activation the int4 output projection reads
// (hip-hoist-dynamic-quant).
//
//   Before:
//     %n  = collapse(hip.rms_norm(expand(%x) : [..., H, D], %w [D]))
//     %nf = hip.cast<f32>(%n)
//     %zf = hip.cast<f32>(%z)
//     %s  = hip.sigmoid(%zf)
//     %g  = hip.mul(%zf, %s)
//     %p  = hip.mul(%nf, %g)                     (or hip.swiglu(%zf, %nf))
//     %y  = hip.cast<f16>(%p)
//   After:
//     %y  = hip.gated_rms_norm ins(%x, %z, %w) {epsilon}
//
// The op reproduces the rounding of every folded kernel (see its
// description), so the rewrite is exact. Every intermediate has the shape of
// %x or of its per-head view, so no product broadcasts and their shape
// queries can be answered by %x.
//
//===----------------------------------------------------------------------===//

#include "gated_rms_norm_fusion.hpp"

#include "fusion_utils.hpp"

namespace hip {
namespace fusion_transform {
namespace {

using namespace utils;
using mlir::hip::CastOp;
using mlir::hip::GatedRmsNormOp;
using mlir::hip::MulOp;
using mlir::hip::RmsNormOp;
using mlir::hip::SigmoidOp;
using mlir::hip::SwigluOp;

// `v` is cast<f32>(src) with `consumer` as its only value use; returns src.
mlir::Value castFromF16(mlir::Value v, mlir::Operation *consumer,
                        llvm::SmallVectorImpl<mlir::Operation *> &chain) {
  auto cast = v.getDefiningOp<CastOp>();
  if (!cast || cast.getTo() != kOnnxFloat || !isF16(cast.getInput()) ||
      (consumer && !hasSingleValueUse(v, consumer)))
    return nullptr;
  chain.push_back(cast);
  return cast.getInput();
}

// The normalized operand: a per-head norm or a norm over the last axis.
std::optional<HeadNorm> matchNorm(mlir::Value n, mlir::Operation *consumer,
                                  mlir::Value ctx) {
  if (std::optional<HeadNorm> head = matchHeadNorm(n, consumer, ctx))
    return head;
  auto norm = n.getDefiningOp<RmsNormOp>();
  if (!norm || norm.getCtx() != ctx || !hasSingleValueUse(n, consumer) ||
      !isF16(norm.getInput()))
    return std::nullopt;
  auto type = rankedType(norm.getInput());
  auto weightType = rankedType(norm.getScale());
  if (!type || type.getRank() < 1 || !weightType ||
      weightType.getRank() != 1 || !isF16(norm.getScale()) ||
      (norm.getAxis() != -1 && norm.getAxis() != type.getRank() - 1) ||
      type.isDynamicDim(type.getRank() - 1) ||
      weightType.getDimSize(0) != type.getDimSize(type.getRank() - 1))
    return std::nullopt;
  HeadNorm match{norm.getInput(), norm.getScale(), weightType.getDimSize(0),
                 norm.getEpsilon().convertToFloat(), {}};
  match.chain = {norm};
  return match;
}

struct GatedRmsNormFusion : public mlir::OpRewritePattern<CastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(CastOp out,
                  mlir::PatternRewriter &rewriter) const override {
    if (out.getTo() != kOnnxFloat16 || !isF16(out->getResult(0)))
      return rewriter.notifyMatchFailure(out, "not a cast to f16");
    mlir::Value ctx = out.getCtx();
    mlir::Value product = out.getInput();
    mlir::Operation *productOp = product.getDefiningOp();
    if (!productOp || !hasSingleValueUse(product, out))
      return rewriter.notifyMatchFailure(out, "no gated product");

    // chain: the ops to erase, consumers before producers.
    llvm::SmallVector<mlir::Operation *> chain = {productOp};
    mlir::Value normF32, z;
    if (auto swiglu = mlir::dyn_cast<SwigluOp>(productOp)) {
      if (swiglu.getCtx() != ctx)
        return rewriter.notifyMatchFailure(out, "ops must share a context");
      normF32 = swiglu.getUp();
      z = castFromF16(swiglu.getGate(), swiglu, chain);
    } else if (auto mul = mlir::dyn_cast<MulOp>(productOp)) {
      if (mul.getCtx() != ctx)
        return rewriter.notifyMatchFailure(out, "ops must share a context");
      for (auto [silu, other] : {std::pair{mul.getRhs(), mul.getLhs()},
                                 std::pair{mul.getLhs(), mul.getRhs()}}) {
        auto siluMul = silu.getDefiningOp<MulOp>();
        if (!siluMul || siluMul.getCtx() != ctx ||
            !hasSingleValueUse(silu, mul))
          continue;
        SigmoidOp sigmoid;
        mlir::Value gate;
        if (!matchEither(siluMul.getLhs(), siluMul.getRhs(), sigmoid, gate) ||
            sigmoid.getX() != gate || sigmoid.getCtx() != ctx ||
            !hasSingleValueUse(sigmoid->getResult(0), siluMul))
          continue;
        // The gate cast feeds exactly the sigmoid and the SiLU product.
        auto gateUsedInChain = llvm::all_of(gate.getUsers(), [&](auto *u) {
          return u == sigmoid || u == siluMul ||
                 mlir::isa<mlir::tensor::DimOp>(u);
        });
        if (!gateUsedInChain)
          continue;
        chain.append({siluMul, sigmoid});
        normF32 = other;
        z = castFromF16(gate, nullptr, chain);
        break;
      }
    }
    if (!z)
      return rewriter.notifyMatchFailure(out, "no SiLU of an f16 gate");

    mlir::Value n = castFromF16(normF32, productOp, chain);
    if (!n)
      return rewriter.notifyMatchFailure(out, "no f16 norm operand");
    mlir::Operation *normCast = chain.back();
    std::optional<HeadNorm> norm = matchNorm(n, normCast, ctx);
    if (!norm)
      return rewriter.notifyMatchFailure(out, "operand is not an RMS norm");

    mlir::Value x = norm->input;
    auto xType = rankedType(x);
    const int64_t width = xType.getDimSize(xType.getRank() - 1);
    if (z.getType() != x.getType() ||
        out->getResult(0).getType() != x.getType() ||
        norm->headSize % 2 != 0 || mlir::ShapedType::isDynamic(width) ||
        width % norm->headSize != 0)
      return rewriter.notifyMatchFailure(out, "unsupported shapes");
    // The norm's per-head views were checked by matchNorm.
    for (mlir::Operation *op : chain) {
      auto type = rankedType(op->getResult(0));
      if (!type || type.getShape() != xType.getShape())
        return rewriter.notifyMatchFailure(out, "chain broadcasts");
    }
    chain.append(norm->chain.begin(), norm->chain.end());
    for (mlir::Operation *op : chain)
      if (!shapeQueriesRetargetable(op->getResult(0), x))
        return rewriter.notifyMatchFailure(out, "unexpected shape query");

    for (mlir::Operation *op : chain)
      retargetShapeQueries(rewriter, op->getResult(0), x);
    rewriter.setInsertionPoint(out);
    auto fused = GatedRmsNormOp::create(
        rewriter, out.getLoc(), out->getResultTypes(), ctx, x, z,
        norm->weight, mlir::ValueRange{out.getOutput()},
        rewriter.getF32FloatAttr(norm->epsilon), nullptr);
    rewriter.replaceOp(out, fused->getResults());
    eraseChain(rewriter, chain);
    return mlir::success();
  }
};

} // namespace

void populateGatedRmsNormFusionPattern(mlir::RewritePatternSet &patterns,
                                       mlir::PatternBenefit benefit) {
  patterns.add<GatedRmsNormFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
