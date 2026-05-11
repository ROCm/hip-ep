//===- RotaryEmbeddingConversion.cpp - ONNX-to-HIP RotaryEmbedding conversion -
//*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Custom(RotaryEmbedding) -> hip.rope
struct RotaryEmbeddingToHip : public RewritePattern {
  RotaryEmbeddingToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
RotaryEmbeddingToHip::matchAndRewrite(Operation *op,
                                      PatternRewriter &rewriter) const {
  // Check if this is RotaryEmbedding
  auto funcNameAttr = op->getAttrOfType<StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "RotaryEmbedding")
    return rewriter.notifyMatchFailure(op, "not a RotaryEmbedding operation");

  // Check domain is "com.microsoft"
  auto domainAttr = op->getAttrOfType<StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for RotaryEmbedding");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();

  // Check operands (should be 4: input, position_ids, cos_cache, sin_cache)
  if (op->getNumOperands() != 4)
    return rewriter.notifyMatchFailure(
        op, "expected 4 operands for RotaryEmbedding");

  Value input = op->getOperand(0);
  Value positionIds = op->getOperand(1);
  Value cosCache = op->getOperand(2);
  Value sinCache = op->getOperand(3);

  // All attributes are optional per the ONNX com.microsoft.RotaryEmbedding
  // spec.  Defaults: interleaved=0, num_heads=0, rotary_embedding_dim=0.
  // A value of 0 for num_heads / rotary_embedding_dim means "infer from
  // tensor shapes".
  auto interleavedAttr = op->getAttrOfType<IntegerAttr>("interleaved");
  auto numHeadsAttr = op->getAttrOfType<IntegerAttr>("num_heads");
  auto rotaryDimAttr = op->getAttrOfType<IntegerAttr>("rotary_embedding_dim");

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
    auto cosCacheType = dyn_cast<RankedTensorType>(cosCache.getType());
    if (cosCacheType && cosCacheType.hasStaticShape() &&
        cosCacheType.getRank() >= 2) {
      rotaryDimVal = cosCacheType.getShape().back() * 2;
    } else {
      return rewriter.notifyMatchFailure(
          op, "Cannot infer rotary_embedding_dim: "
              "cos_cache must have static shape with rank >= 2");
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
    if (inputType && inputType.hasStaticShape() && inputType.getRank() >= 1) {
      int64_t hidden = inputType.getShape().back();
      numHeadsVal = hidden / rotaryDimVal;
    } else {
      return rewriter.notifyMatchFailure(op, "Cannot infer num_heads: "
                                             "input must have static shape");
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

  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

  // Create init tensor
  Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Create hip.rope operation
  auto hipOp = mlir::hip::RopeOp::create(
      rewriter, loc, resultType, context, input, positionIds, cosCache,
      sinCache, init, interleavedI64Attr, numHeadsI64Attr, rotaryDimI64Attr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

} // namespace

void mlir::hip::populateRotaryEmbeddingConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<RotaryEmbeddingToHip>(ctx);
}

} // namespace hip
} // namespace mlir
