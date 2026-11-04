/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- gqa_rope_fold.cpp - fold hip.rope on Q/K into hip.gqa --------------===//
//
// A GroupQueryAttention export with do_rotary = 0 applies RoPE through two
// separate RotaryEmbedding nodes. hip.gqa already rotates Q and K itself when
// do_rotary = 1, so the separate ropes cost two kernel launches per layer and
// a round trip of Q and K through memory. The fold hands the rotation to the
// attention op with the same position_ids, so positions stay exactly what the
// export computed (left padding or packed batches included) rather than being
// re-derived from seqlens_k.
//
//   Before:
//     %q = hip.rope(%ctx) ins(%q_in, %pos, %cos, %sin : ...) outs(%e0 : t)
//         {interleaved = 0, num_heads = H, rotary_embedding_dim = D} : t
//     %k = hip.rope(%ctx) ins(%k_in, %pos, %cos, %sin : ...) outs(%e1 : tk)
//         {interleaved = 0, num_heads = G, rotary_embedding_dim = D} : tk
//     %r:3 = hip.gqa(%ctx) ins(%q, %k, %v, %pk, %pv, %sl, %tot : ...)
//         outs(...) {do_rotary = 0, num_heads = H, kv_num_heads = G}
//   After:
//     %r:3 = hip.gqa(%ctx) ins(%q_in, %k_in, %v, %pk, %pv, %sl, %tot, %cos,
//                              %sin, %pos : ...)
//         outs(...) {do_rotary = 1, num_heads = H, kv_num_heads = G}
//
// The attention op rotates in fp32 and rounds once; hip.rope on fp16 rounds
// the products, so results move by at most an fp16 ulp per element.
//
//===----------------------------------------------------------------------===//

#include "gqa_rope_fold.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace hip {
namespace fusion_transform {
namespace {

// True when every use is either `consumer` (once) or a tensor.dim, which
// only reads the type; the rope result has its input's type.
bool hasSingleValueUse(mlir::Value value, mlir::Operation *consumer) {
  mlir::Operation *seen = nullptr;
  for (mlir::OpOperand &use : value.getUses()) {
    if (mlir::isa<mlir::tensor::DimOp>(use.getOwner()))
      continue;
    if (seen)
      return false;
    seen = use.getOwner();
  }
  return seen == consumer;
}

struct GqaRopeFold : public mlir::OpRewritePattern<mlir::hip::GqaOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::GqaOp gqa,
                  mlir::PatternRewriter &rewriter) const override {
    if (gqa.getDoRotary() != 0 || gqa.getCosCache() || gqa.getSinCache() ||
        gqa.getPositionIds())
      return rewriter.notifyMatchFailure(gqa, "attention already rotates");
    if (!gqa.getKey() || !gqa.getValue())
      return rewriter.notifyMatchFailure(gqa, "packed QKV has no rope inputs");
    if (gqa.getNoCausal())
      return rewriter.notifyMatchFailure(gqa, "bidirectional attention");

    auto qRope = gqa.getQuery().getDefiningOp<mlir::hip::RopeOp>();
    auto kRope = gqa.getKey().getDefiningOp<mlir::hip::RopeOp>();
    if (!qRope || !kRope || qRope == kRope)
      return rewriter.notifyMatchFailure(gqa, "query/key are not hip.rope");
    if (qRope.getCtx() != gqa.getCtx() || kRope.getCtx() != gqa.getCtx())
      return rewriter.notifyMatchFailure(gqa, "ops must share a context");

    mlir::Value pos = qRope.getPositionIds();
    mlir::Value cos = qRope.getCosCache();
    mlir::Value sin = qRope.getSinCache();
    if (!pos || kRope.getPositionIds() != pos || kRope.getCosCache() != cos ||
        kRope.getSinCache() != sin)
      return rewriter.notifyMatchFailure(
          gqa, "Q and K must rotate by the same position_ids and caches");
    if (qRope.getInterleaved() != 0 || kRope.getInterleaved() != 0)
      return rewriter.notifyMatchFailure(gqa, "interleaved rope");

    auto ranked = [](mlir::Value v) {
      return mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    };
    auto qType = ranked(qRope.getInput());
    auto kType = ranked(kRope.getInput());
    // fp32 attention over a quantized cache runs through a cast adapter that
    // only converts the cache rows [0, total_seq); fp16 has no such limit.
    if (!qType || !qType.getElementType().isF16())
      return rewriter.notifyMatchFailure(gqa, "only fp16 attention");
    if (!kType || qType.getRank() != 3 || kType.getRank() != 3 ||
        qType.isDynamicDim(2) || kType.isDynamicDim(2) ||
        ranked(qRope->getResult(0)) != qType ||
        ranked(kRope->getResult(0)) != kType)
      return rewriter.notifyMatchFailure(
          gqa, "rope must map a rank-3 [B, S, heads * D] tensor to itself");

    const int64_t numHeads = gqa.getNumHeads();
    const int64_t kvNumHeads = gqa.getKvNumHeads();
    if (numHeads <= 0 || kvNumHeads <= 0 || qType.getDimSize(2) % numHeads)
      return rewriter.notifyMatchFailure(gqa, "head count does not divide");
    const int64_t headSize = qType.getDimSize(2) / numHeads;
    if (kType.getDimSize(2) != kvNumHeads * headSize ||
        static_cast<int64_t>(qRope.getNumHeads()) != numHeads ||
        static_cast<int64_t>(kRope.getNumHeads()) != kvNumHeads)
      return rewriter.notifyMatchFailure(gqa, "rope heads differ from GQA");
    if (static_cast<int64_t>(qRope.getRotaryEmbeddingDim()) != headSize ||
        static_cast<int64_t>(kRope.getRotaryEmbeddingDim()) != headSize)
      return rewriter.notifyMatchFailure(gqa, "partial rotary");

    auto cosType = ranked(cos);
    auto posType = ranked(pos);
    if (!cosType || ranked(sin) != cosType || cosType.getRank() != 2 ||
        cosType.getDimSize(1) != headSize / 2 ||
        cosType.getElementType() != qType.getElementType() ||
        kType.getElementType() != qType.getElementType())
      return rewriter.notifyMatchFailure(
          gqa, "caches must be [max_position, D / 2] in the query type");
    if (!posType || posType.getRank() != 2 ||
        !posType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(gqa,
                                         "position_ids must be i64 [B, S]");

    if (!hasSingleValueUse(qRope->getResult(0), gqa) ||
        !hasSingleValueUse(kRope->getResult(0), gqa))
      return rewriter.notifyMatchFailure(gqa, "rope result has other uses");

    mlir::Value qIn = qRope.getInput();
    mlir::Value kIn = kRope.getInput();
    for (auto [rope, in] : {std::pair{qRope, qIn}, std::pair{kRope, kIn}})
      rewriter.replaceUsesWithIf(
          rope->getResult(0), in, [](mlir::OpOperand &operand) {
            return mlir::isa<mlir::tensor::DimOp>(operand.getOwner());
          });

    rewriter.modifyOpInPlace(gqa, [&] {
      gqa.getQueryMutable().assign(qIn);
      gqa.getKeyMutable().assign(kIn);
      gqa.getCosCacheMutable().assign(cos);
      gqa.getSinCacheMutable().assign(sin);
      gqa.getPositionIdsMutable().assign(pos);
      gqa.setDoRotaryAttr(rewriter.getI64IntegerAttr(1));
      gqa.setRotaryInterleavedAttr(rewriter.getI64IntegerAttr(0));
    });

    for (mlir::hip::RopeOp rope : {qRope, kRope}) {
      mlir::Operation *initDef = rope.getOutput().getDefiningOp();
      rewriter.eraseOp(rope);
      if (initDef && mlir::isOpTriviallyDead(initDef))
        rewriter.eraseOp(initDef);
    }
    return mlir::success();
  }
};

} // namespace

void populateGqaRopeFoldPattern(mlir::RewritePatternSet &patterns,
                                mlir::PatternBenefit benefit) {
  patterns.add<GqaRopeFold>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
