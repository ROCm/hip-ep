/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Native ai.onnx RotaryEmbedding (opset >= 23) -> hip.rope
///
/// Distinct from com.microsoft.RotaryEmbedding (handled in
/// RotaryEmbeddingConversion.cpp as an onnx.Custom op). The standard op differs
/// in two ways this pattern handles:
///   1. Operand order is (X, cos_cache, sin_cache, position_ids?) -- cos/sin
///      come before the optional position_ids, whereas the contrib op is
///      (input, position_ids, cos_cache, sin_cache).
///   2. position_ids is OPTIONAL. When omitted, cos_cache/sin_cache are already
///      position-expanded to [batch, seq, rotary_dim/2] (HF exports precompute
///      the per-token cos/sin via a Gather). hip.rope with an absent
///      position_ids operand indexes them directly by the flat token position.
///
/// Before (Gemma-style, no position_ids, precomputed 3D cos/sin):
///   %y = onnx.RotaryEmbedding(%x, %cos, %sin)
///        {interleaved = 0, num_heads = 16, rotary_embedding_dim = 0}
///        : (tensor<?x?x4096xf16>, tensor<?x?x128xf16>, tensor<?x?x128xf16>)
///          -> tensor<?x?x4096xf16>
/// After:
///   %init = tensor.empty(...) : tensor<?x?x4096xf16>
///   %y = hip.rope(%ctx) ins(%x, %cos, %sin : ...) outs(%init : ...)
///        {interleaved = 0, num_heads = 16, rotary_embedding_dim = 128}
struct OnnxRotaryEmbeddingToHip : public mlir::RewritePattern {
  OnnxRotaryEmbeddingToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.RotaryEmbedding", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult OnnxRotaryEmbeddingToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op->getLoc();

  size_t numOps = op->getNumOperands();
  if (numOps < 3 || numOps > 4)
    return rewriter.notifyMatchFailure(
        op, "onnx.RotaryEmbedding expects 3-4 operands "
            "(X, cos_cache, sin_cache, [position_ids])");

  auto isLive = [](mlir::Value v) {
    return v && !mlir::isa<mlir::NoneType>(v.getType());
  };

  mlir::Value input = op->getOperand(0);
  mlir::Value cosCache = op->getOperand(1);
  mlir::Value sinCache = op->getOperand(2);
  mlir::Value positionIds = numOps > 3 ? op->getOperand(3) : mlir::Value();
  if (positionIds && !isLive(positionIds))
    positionIds = nullptr;

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  // Both ONNX spellings share attribute defaults and shape-based inference.
  mlir::FailureOr<RotaryEmbeddingConfig> config =
      resolveRotaryEmbeddingConfig(rewriter, op, input, cosCache);
  if (mlir::failed(config))
    return mlir::failure();

  if (config->numHeads <= 0)
    return rewriter.notifyMatchFailure(op,
                                       "RotaryEmbedding: invalid num_heads");

  auto interleavedI64Attr = rewriter.getI64IntegerAttr(config->interleaved);
  auto numHeadsI64Attr = rewriter.getI64IntegerAttr(config->numHeads);
  auto rotaryDimI64Attr = rewriter.getI64IntegerAttr(config->rotaryDim);

  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "expected 1 result for RotaryEmbedding");

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // hip.rope operand order is (input, position_ids, cos_cache, sin_cache);
  // positionIds is a null Value when the ONNX op omitted it (the Optional
  // operand is then absent and the runtime uses flat-token indexing).
  auto hipOp = mlir::hip::RopeOp::create(
      rewriter, loc, context, input, positionIds, cosCache, sinCache, init,
      interleavedI64Attr, numHeadsI64Attr, rotaryDimI64Attr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateOnnxRotaryEmbeddingConversionPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx) {
  patterns.add<OnnxRotaryEmbeddingToHip>(ctx);
}

} // namespace hip
} // namespace mlir
