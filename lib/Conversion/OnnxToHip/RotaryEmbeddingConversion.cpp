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

} // namespace

void populateRotaryEmbeddingConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<RotaryEmbeddingToHip>(ctx);
}

} // namespace hip
} // namespace mlir
