/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Custom(RotaryEmbedding) -> hip.rope
struct RotaryEmbeddingToHip : public mlir::RewritePattern {
  RotaryEmbeddingToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
RotaryEmbeddingToHip::matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const {
  // Check if this is RotaryEmbedding
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "RotaryEmbedding")
    return rewriter.notifyMatchFailure(op, "not a RotaryEmbedding operation");

  // Check domain is "com.microsoft"
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for RotaryEmbedding");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // Check operands (should be 4: input, position_ids, cos_cache, sin_cache)
  if (op->getNumOperands() != 4)
    return rewriter.notifyMatchFailure(
        op, "expected 4 operands for RotaryEmbedding");

  mlir::Value input = op->getOperand(0);
  mlir::Value positionIds = op->getOperand(1);
  mlir::Value cosCache = op->getOperand(2);
  mlir::Value sinCache = op->getOperand(3);

  // All attributes are optional per the ONNX com.microsoft.RotaryEmbedding
  // spec.  Defaults: interleaved=0, num_heads=0, rotary_embedding_dim=0.
  // A value of 0 for num_heads / rotary_embedding_dim means "infer from
  // tensor shapes".
  auto interleavedAttr = op->getAttrOfType<mlir::IntegerAttr>("interleaved");
  auto numHeadsAttr = op->getAttrOfType<mlir::IntegerAttr>("num_heads");
  auto rotaryDimAttr =
      op->getAttrOfType<mlir::IntegerAttr>("rotary_embedding_dim");

  int64_t interleavedVal = interleavedAttr ? interleavedAttr.getSInt() : 0;
  int64_t numHeadsVal = numHeadsAttr ? numHeadsAttr.getSInt() : 0;
  int64_t rotaryDimVal = rotaryDimAttr ? rotaryDimAttr.getSInt() : 0;

  // ONNX com.microsoft.RotaryEmbedding: 0 means "infer from tensor shapes"
  //
  //   cos_cache: [max_seq, rotary_dim/2] -> rotary_dim = last_dim * 2
  //
  //   Input shape options (per ONNX spec):
  //     3D [batch, seq, hidden]           -> num_heads = hidden / rotary_dim
  //                                          head_dim = hidden / num_heads
  //                                          rotary_dim defaults to head_dim
  //                                          when not provided
  //     4D [batch, num_heads, seq, head]  -> num_heads = shape[1]
  //                                          head_dim = shape[3]
  //                                          rotary_dim defaults to head_dim
  //                                          when not provided
  //
  // NOTE: When rotary_embedding_dim attribute is provided on a 3D input it is
  // legal for rotary_dim < head_dim (M-RoPE / partial rotary).  In that case
  // num_heads CANNOT be derived as hidden/rotary_dim and must come from the
  // num_heads attribute.
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());

  if (rotaryDimVal == 0) {
    auto cosCacheType =
        mlir::dyn_cast<mlir::RankedTensorType>(cosCache.getType());
    // Only the last dim (rotary_dim/2) is needed — it's always a static
    // architecture constant even when batch/seq dims are dynamic.
    if (cosCacheType && cosCacheType.getRank() >= 2 &&
        !cosCacheType.isDynamicDim(cosCacheType.getRank() - 1)) {
      rotaryDimVal = cosCacheType.getDimSize(cosCacheType.getRank() - 1) * 2;
    } else {
      return rewriter.notifyMatchFailure(
          op, "Cannot infer rotary_embedding_dim: "
              "cos_cache last dim must be static with rank >= 2");
    }
  }

  if (inputType && inputType.getRank() == 4) {
    int64_t shapeNumHeads = inputType.getShape()[1];
    if (shapeNumHeads != mlir::ShapedType::kDynamic) {
      if (numHeadsVal == 0) {
        numHeadsVal = shapeNumHeads;
      } else if (numHeadsVal != shapeNumHeads) {
        return rewriter.notifyMatchFailure(
            op, "RotaryEmbedding: num_heads attribute disagrees with 4D "
                "input shape (BNSH)");
      }
    }
    if (numHeadsVal == 0)
      return rewriter.notifyMatchFailure(
          op, "Cannot infer num_heads: 4D input has dynamic num_heads dim "
              "and no num_heads attribute");
  } else if (numHeadsVal == 0 && rotaryDimVal > 0) {
    // Only the last dim (hidden_size) is needed — it's always a static
    // architecture constant even when batch/seq dims are dynamic.
    if (inputType && inputType.getRank() >= 1 &&
        !inputType.isDynamicDim(inputType.getRank() - 1)) {
      int64_t hidden = inputType.getDimSize(inputType.getRank() - 1);
      numHeadsVal = hidden / rotaryDimVal;
    } else {
      return rewriter.notifyMatchFailure(
          op, "Cannot infer num_heads: input last dim must be static");
    }
  }

  // Convert to i64 attributes (using inferred or original values)
  auto interleavedI64Attr = rewriter.getI64IntegerAttr(interleavedVal);
  auto numHeadsI64Attr = rewriter.getI64IntegerAttr(numHeadsVal);
  auto rotaryDimI64Attr = rewriter.getI64IntegerAttr(rotaryDimVal);

  // Should have 1 result: output
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "expected 1 result for RotaryEmbedding");

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Create init tensor
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Result type inferred from `init` via InferTypeOpInterface — DPS contract:
  // result type == outs operand type.
  auto hipOp = mlir::hip::RopeOp::create(
      rewriter, loc, context, input, positionIds, cosCache, sinCache, init,
      interleavedI64Attr, numHeadsI64Attr, rotaryDimI64Attr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// com.microsoft.MRotaryEmbedding -> hip.mrotary_embedding
struct MRotaryEmbeddingToHip : public mlir::RewritePattern {
  MRotaryEmbeddingToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MRotaryEmbeddingToHip::matchAndRewrite(mlir::Operation *op,
                                       mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "MRotaryEmbedding") {
    return rewriter.notifyMatchFailure(op, "not a MRotaryEmbedding operation");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for MRotaryEmbedding");
  }
  if (op->getNumOperands() != 4 || op->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(
        op, "MRotaryEmbedding expects 4 operands and 1 result");
  }

  auto readInt = [&](llvm::StringRef name, int64_t defaultVal) {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    return attr ? attr.getValue().getSExtValue() : defaultVal;
  };
  if (readInt("interleaved", 0) != 0) {
    return rewriter.notifyMatchFailure(op, "interleaved must be 0");
  }
  if (readInt("is_packed_batching", 0) != 0) {
    return rewriter.notifyMatchFailure(op, "is_packed_batching must be 0");
  }
  const int64_t layout = readInt("mrope_layout", 0);
  if (layout != 0 && layout != 1) {
    return rewriter.notifyMatchFailure(op, "mrope_layout must be 0 or 1");
  }
  float scale = 1.0f;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("scale")) {
    scale = static_cast<float>(attr.getValueAsDouble());
  }

  mlir::Value input = op->getOperand(0);
  mlir::Value positionIds = op->getOperand(1);
  mlir::Value cosCache = op->getOperand(2);
  mlir::Value sinCache = op->getOperand(3);

  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() != 3 || inputType.isDynamicDim(2)) {
    return rewriter.notifyMatchFailure(
        op, "input must be rank 3 with a static hidden dim");
  }
  mlir::Type elem = inputType.getElementType();
  if (!elem.isF16() && !elem.isF32()) {
    return rewriter.notifyMatchFailure(op, "input must be f16 or f32");
  }
  const int64_t hidden = inputType.getDimSize(2);
  const int64_t numHeads = readInt("num_heads", 0);
  if (numHeads <= 0 || hidden % numHeads != 0) {
    return rewriter.notifyMatchFailure(
        op, "num_heads must be positive and divide the hidden dim");
  }
  const int64_t headSize = hidden / numHeads;
  int64_t rotaryDim = readInt("rotary_embedding_dim", 0);
  if (rotaryDim == 0) {
    rotaryDim = headSize;
  }
  if (rotaryDim <= 0 || rotaryDim % 2 != 0 || rotaryDim > headSize) {
    return rewriter.notifyMatchFailure(
        op, "rotary_embedding_dim must be positive, even and <= head size");
  }
  const int64_t half = rotaryDim / 2;

  auto sectionAttr = op->getAttrOfType<mlir::ArrayAttr>("mrope_section");
  if (!sectionAttr || sectionAttr.size() != 3) {
    return rewriter.notifyMatchFailure(op,
                                       "mrope_section must have 3 elements");
  }
  llvm::SmallVector<int64_t, 3> section;
  for (mlir::Attribute a : sectionAttr) {
    auto i = mlir::dyn_cast<mlir::IntegerAttr>(a);
    if (!i || i.getValue().getSExtValue() < 0) {
      return rewriter.notifyMatchFailure(
          op, "mrope_section values must be non-negative integers");
    }
    section.push_back(i.getValue().getSExtValue());
  }
  if (section[0] + section[1] + section[2] != half) {
    return rewriter.notifyMatchFailure(
        op, "mrope_section must sum to rotary_embedding_dim / 2");
  }

  auto dimMismatch = [](int64_t got, int64_t want) {
    return !mlir::ShapedType::isDynamic(got) &&
           !mlir::ShapedType::isDynamic(want) && got != want;
  };
  auto posType = mlir::dyn_cast<mlir::RankedTensorType>(positionIds.getType());
  if (!posType || !posType.getElementType().isInteger(64) ||
      posType.getRank() != 3 || posType.getDimSize(0) != 3 ||
      dimMismatch(posType.getDimSize(1), inputType.getDimSize(0)) ||
      dimMismatch(posType.getDimSize(2), inputType.getDimSize(1))) {
    return rewriter.notifyMatchFailure(op,
                                       "position_ids must be i64 [3, B, S]");
  }
  for (mlir::Value cache : {cosCache, sinCache}) {
    auto cacheType = mlir::dyn_cast<mlir::RankedTensorType>(cache.getType());
    if (!cacheType || cacheType.getElementType() != elem ||
        cacheType.getRank() != 2 || cacheType.getDimSize(1) != half) {
      return rewriter.notifyMatchFailure(
          op, "cos_cache / sin_cache must be [max_position, "
              "rotary_embedding_dim / 2] with the input element type");
    }
  }
  if (dimMismatch(
          mlir::cast<mlir::ShapedType>(cosCache.getType()).getDimSize(0),
          mlir::cast<mlir::ShapedType>(sinCache.getType()).getDimSize(0))) {
    return rewriter.notifyMatchFailure(
        op, "cos_cache and sin_cache must have the same shape");
  }

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType.getElementType() != elem ||
      resultType.getRank() != 3 ||
      dimMismatch(resultType.getDimSize(0), inputType.getDimSize(0)) ||
      dimMismatch(resultType.getDimSize(1), inputType.getDimSize(1)) ||
      dimMismatch(resultType.getDimSize(2), hidden)) {
    return rewriter.notifyMatchFailure(op, "output must match the input");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "missing context argument");
  }

  mlir::Location loc = op->getLoc();
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::MRotaryEmbeddingOp::create(
      rewriter, loc, *ctxOrFailure, input, positionIds, cosCache, sinCache,
      init, rewriter.getI64IntegerAttr(numHeads),
      rewriter.getI64IntegerAttr(rotaryDim), rewriter.getI64ArrayAttr(section),
      rewriter.getI64IntegerAttr(layout), rewriter.getF32FloatAttr(scale));
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateRotaryEmbeddingConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<RotaryEmbeddingToHip, MRotaryEmbeddingToHip>(ctx);
}

} // namespace hip
} // namespace mlir
