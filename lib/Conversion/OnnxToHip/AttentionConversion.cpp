/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Custom(com.microsoft.Attention) -> hip.gqa.
///
/// The legacy fused-QKV Attention node (Whisper encoder, CLIP text encoder,
/// BERT, diffusion mid-blocks) is rewritten to:
///
///   1. One QKV projection over the fused weight (no transpose):
///        qkv_proj   = hip.matmul(hidden, qkv_w)   : [B,S,H] @ [H,3H]
///        qkv_biased = hip.add(qkv_proj, qkv_b)
///      ORT stores the fused weight as [H, q+k+v]. When qkv_hidden_sizes is
///      absent and that weight is static [H, 3H], the three sizes are inferred.
///
///   2. Three tensor.extract_slice ops splitting the last axis into Q / K / V.
///
///   3. One hip.gqa. unidirectional=0 is bidirectional (no_causal).
///      unidirectional=1 is causal. Batch and sequence may be dynamic. A
///      rank-1 length or rank-2 padding mask_index is expanded into
///      attention_bias. Past KV, an already-built attention_bias operand, and
///      past_sequence_length stay unsupported.
///
/// Before:
///   %out = onnx.Custom(%hidden, %qkv_w, %qkv_b) {function_name = "Attention",
///            domain_name = "com.microsoft", num_heads = 1}
///   // %qkv_w : tensor<512x1536xf16>, hidden batch/sequence dynamic
/// After:
///   %qkv = hip.matmul %ctx, %hidden, %qkv_w
///   %biased = hip.add %ctx, %qkv, %qkv_b
///   %q = tensor.extract_slice %biased[0, 0, 0][batch, seq, 512][1, 1, 1]
///   ...
///   %out = hip.gqa(%ctx) ins(%q, %k, %v, %seqlens, %total_seq, ...)
///            {no_causal = true, num_heads = 1, kv_num_heads = 1}
struct AttentionToHip : public mlir::RewritePattern {
  AttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Rank-1 mask_index is a per-batch count of valid leading tokens. Rank-2 is a
/// 0/1 token mask (nonzero = keep). hip.gqa indexes attention_bias as
/// [batch, heads, query, key] and does not broadcast a query extent of 1, so
/// the key mask is repeated on every query row. heads is 1 and does broadcast.
mlir::FailureOr<mlir::Value>
buildPaddingBias(mlir::PatternRewriter &rewriter, mlir::Location loc,
                 mlir::Value hidden, mlir::RankedTensorType hiddenType,
                 mlir::Value mask, int64_t maskRank, mlir::Type biasElem,
                 double filter) {
  const int64_t batch = hiddenType.getDimSize(0);
  const int64_t seqLen = hiddenType.getDimSize(1);
  auto biasType =
      mlir::RankedTensorType::get({batch, 1, seqLen, seqLen}, biasElem);
  llvm::SmallVector<mlir::Value> dynSizes;
  if (hiddenType.isDynamicDim(0))
    dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, hidden, 0));
  if (hiddenType.isDynamicDim(1)) {
    mlir::Value seq = mlir::tensor::DimOp::create(rewriter, loc, hidden, 1);
    dynSizes.push_back(seq);
    dynSizes.push_back(seq);
  }

  auto gen = mlir::tensor::GenerateOp::create(
      rewriter, loc, biasType, dynSizes,
      [&](mlir::OpBuilder &b, mlir::Location bodyLoc, mlir::ValueRange ivs) {
        mlir::Value batchIdx = ivs[0];
        mlir::Value keyIdx = ivs[3];
        mlir::Value keep;
        if (maskRank == 1) {
          mlir::Value len = mlir::tensor::ExtractOp::create(
                                b, bodyLoc, mask, mlir::ValueRange{batchIdx})
                                .getResult();
          mlir::Value keyInt = mlir::arith::IndexCastOp::create(
              b, bodyLoc, len.getType(), keyIdx);
          keep = mlir::arith::CmpIOp::create(
              b, bodyLoc, mlir::arith::CmpIPredicate::slt, keyInt, len);
        } else {
          mlir::Value bit =
              mlir::tensor::ExtractOp::create(
                  b, bodyLoc, mask, mlir::ValueRange{batchIdx, keyIdx})
                  .getResult();
          mlir::Value zero = mlir::arith::ConstantOp::create(
              b, bodyLoc, b.getIntegerAttr(bit.getType(), 0));
          keep = mlir::arith::CmpIOp::create(
              b, bodyLoc, mlir::arith::CmpIPredicate::ne, bit, zero);
        }
        mlir::Value accepted = mlir::arith::ConstantOp::create(
            b, bodyLoc, b.getFloatAttr(biasElem, 0.0));
        mlir::Value rejected = mlir::arith::ConstantOp::create(
            b, bodyLoc, b.getFloatAttr(biasElem, filter));
        mlir::Value chosen =
            mlir::arith::SelectOp::create(b, bodyLoc, keep, accepted, rejected);
        mlir::tensor::YieldOp::create(b, bodyLoc, chosen);
      });
  return gen.getResult();
}

mlir::LogicalResult
AttentionToHip::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "Attention")
    return rewriter.notifyMatchFailure(op, "not an Attention operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for Attention");

  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "Attention with present outputs is not lowered");
  if (op->getNumOperands() < 3)
    return rewriter.notifyMatchFailure(
        op, "Attention expects at least 3 operands (input, weights, bias)");

  auto operandOrNull = [&](size_t index) -> mlir::Value {
    if (index >= op->getNumOperands())
      return {};
    mlir::Value value = op->getOperand(index);
    if (!value || mlir::isa<mlir::NoneType>(value.getType()))
      return {};
    return value;
  };
  // Operands: hidden, weight, bias, mask_index, past, attention_bias,
  // past_sequence_length. Only mask_index is lowered.
  mlir::Value mask = operandOrNull(3);
  for (size_t i = 4; i < op->getNumOperands(); ++i) {
    if (operandOrNull(i))
      return rewriter.notifyMatchFailure(
          op, "unsupported optional Attention operand (past / attention_bias "
              "/ past_seqlen)");
  }

  mlir::Value hidden = op->getOperand(0);
  mlir::Value qkvW = op->getOperand(1);
  mlir::Value qkvB = op->getOperand(2);

  auto numHeadsAttrOnnx = op->getAttrOfType<mlir::IntegerAttr>("num_heads");
  if (!numHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing num_heads attribute");
  const int64_t numHeads = numHeadsAttrOnnx.getValue().getSExtValue();
  if (numHeads <= 0)
    return rewriter.notifyMatchFailure(op, "num_heads must be > 0");

  auto getI64 = [&](const char *name, int64_t defaultVal) {
    auto a = op->getAttrOfType<mlir::IntegerAttr>(name);
    return a ? a.getValue().getSExtValue() : defaultVal;
  };
  if (getI64("do_rotary", 0) != 0)
    return rewriter.notifyMatchFailure(op, "do_rotary != 0 not supported");
  const int64_t unidirectional = getI64("unidirectional", 0);
  if (unidirectional != 0 && unidirectional != 1)
    return rewriter.notifyMatchFailure(
        op, "unidirectional must be 0 (bidirectional) or 1 (causal)");
  if (getI64("past_present_share_buffer", 0) != 0)
    return rewriter.notifyMatchFailure(
        op, "past_present_share_buffer != 0 not supported");
  // unidirectional=0 skips the runtime causal triangle.
  const bool noCausal = unidirectional == 0;

  // Scale: ONNX attr if present, else 0.0 so hip.gqa computes
  // 1/sqrt(head_size).
  auto scaleAttrOnnx = op->getAttrOfType<mlir::FloatAttr>("scale");
  const float scale =
      scaleAttrOnnx ? scaleAttrOnnx.getValue().convertToFloat() : 0.0f;

  auto hiddenType = mlir::dyn_cast<mlir::RankedTensorType>(hidden.getType());
  if (!hiddenType || hiddenType.getRank() != 3)
    return rewriter.notifyMatchFailure(
        op, "hidden must be a rank-3 tensor [B, S, H]");
  auto qkvWType = mlir::dyn_cast<mlir::RankedTensorType>(qkvW.getType());
  if (!qkvWType || qkvWType.getRank() != 2 || !qkvWType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "qkv_w must be a static rank-2 tensor [H, 3H]");
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!outputType || outputType.getRank() != 3)
    return rewriter.notifyMatchFailure(
        op, "Attention output must be a rank-3 tensor");

  const int64_t weightIn = qkvWType.getDimSize(0);
  const int64_t weightOut = qkvWType.getDimSize(1);
  int64_t qSize = 0;
  auto qkvSizesAttr = op->getAttrOfType<mlir::ArrayAttr>("qkv_hidden_sizes");
  if (qkvSizesAttr) {
    if (qkvSizesAttr.size() != 3)
      return rewriter.notifyMatchFailure(
          op, "qkv_hidden_sizes must be a 3-element array");
    llvm::SmallVector<int64_t, 3> qkvSizes;
    for (mlir::Attribute a : qkvSizesAttr) {
      auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
      if (!ia)
        return rewriter.notifyMatchFailure(
            op, "qkv_hidden_sizes entries must be integers");
      qkvSizes.push_back(ia.getValue().getSExtValue());
    }
    if (qkvSizes[0] != qkvSizes[1] || qkvSizes[1] != qkvSizes[2])
      return rewriter.notifyMatchFailure(
          op, "qkv_hidden_sizes entries must be equal (multi-head, HPG=1)");
    qSize = qkvSizes[0];
  } else if (weightOut % 3 == 0 && weightOut / 3 == weightIn) {
    qSize = weightOut / 3;
  } else {
    return rewriter.notifyMatchFailure(
        op, "qkv_hidden_sizes is absent and qkv_w is not static [H, 3H]");
  }

  const int64_t qkvHidden = qSize * 3;
  if (!hiddenType.isDynamicDim(2) && hiddenType.getDimSize(2) != weightIn)
    return rewriter.notifyMatchFailure(op,
                                       "hidden size must match qkv_w.shape[0]");
  if (weightOut != qkvHidden)
    return rewriter.notifyMatchFailure(
        op, "qkv_w shape must be [hidden, sum(qkv_hidden_sizes)]");
  if (qSize % numHeads != 0)
    return rewriter.notifyMatchFailure(
        op, "qkv_hidden_sizes[0] must be divisible by num_heads");

  auto dimsMatch = [](int64_t lhs, int64_t rhs) {
    return lhs == rhs || mlir::ShapedType::isDynamic(lhs) ||
           mlir::ShapedType::isDynamic(rhs);
  };
  if (!dimsMatch(outputType.getDimSize(0), hiddenType.getDimSize(0)) ||
      !dimsMatch(outputType.getDimSize(1), hiddenType.getDimSize(1)) ||
      !dimsMatch(outputType.getDimSize(2), hiddenType.getDimSize(2)))
    return rewriter.notifyMatchFailure(
        op, "Attention output shape must match input hidden shape");

  mlir::Type elemType = hiddenType.getElementType();
  mlir::Value attentionBias;
  if (mask) {
    auto maskType = mlir::dyn_cast<mlir::RankedTensorType>(mask.getType());
    if (!maskType || !maskType.getElementType().isInteger())
      return rewriter.notifyMatchFailure(
          op, "mask_index must be a ranked integer tensor");
    if (maskType.getRank() != 1 && maskType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "mask_index must be rank 1 (lengths) or rank 2 (token mask)");
    if (maskType.getRank() == 1 && maskType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(
          op, "rank-1 mask_index must be a token count, not i1");
    if (!dimsMatch(maskType.getDimSize(0), hiddenType.getDimSize(0)) ||
        (maskType.getRank() == 2 &&
         !dimsMatch(maskType.getDimSize(1), hiddenType.getDimSize(1))))
      return rewriter.notifyMatchFailure(
          op, "mask_index shape does not match hidden batch/sequence");
    if (!elemType.isF16() && !elemType.isF32())
      return rewriter.notifyMatchFailure(
          op, "padding mask requires f16 or f32 hidden");
    double filter = -10000.0;
    if (auto filterAttr =
            op->getAttrOfType<mlir::FloatAttr>("mask_filter_value"))
      filter = filterAttr.getValue().convertToDouble();
    auto biasOr = buildPaddingBias(rewriter, op->getLoc(), hidden, hiddenType,
                                   mask, maskType.getRank(), elemType, filter);
    if (mlir::failed(biasOr))
      return mlir::failure();
    attentionBias = *biasOr;
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  const int64_t batch = hiddenType.getDimSize(0);
  const int64_t seqLen = hiddenType.getDimSize(1);

  mlir::Value batchDyn, seqDyn;
  if (hiddenType.isDynamicDim(0))
    batchDyn = mlir::tensor::DimOp::create(rewriter, loc, hidden, 0);
  if (hiddenType.isDynamicDim(1))
    seqDyn = mlir::tensor::DimOp::create(rewriter, loc, hidden, 1);
  llvm::SmallVector<mlir::Value> prefixDyn;
  if (batchDyn)
    prefixDyn.push_back(batchDyn);
  if (seqDyn)
    prefixDyn.push_back(seqDyn);

  // The fused weight is already [H, 3H], so hip.matmul consumes it directly.
  auto projType =
      mlir::RankedTensorType::get({batch, seqLen, qkvHidden}, elemType);
  mlir::Value projInit =
      mlir::tensor::EmptyOp::create(rewriter, loc, projType, prefixDyn);
  mlir::Value qkvProj =
      mlir::hip::MatmulOp::create(rewriter, loc, projType, context, hidden,
                                  qkvW, projInit)
          .getResult(0);

  mlir::Value addInit =
      mlir::tensor::EmptyOp::create(rewriter, loc, projType, prefixDyn);
  mlir::Value qkvBiased =
      mlir::hip::AddOp::create(rewriter, loc, projType, context, qkvProj, qkvB,
                               addInit)
          .getResult(0);

  auto sliceType =
      mlir::RankedTensorType::get({batch, seqLen, qSize}, elemType);
  auto foldDim = [&](mlir::Value dyn,
                     int64_t staticSize) -> mlir::OpFoldResult {
    if (dyn)
      return dyn;
    return rewriter.getIndexAttr(staticSize);
  };
  auto buildSlice = [&](int64_t offsetOnLastAxis) -> mlir::Value {
    llvm::SmallVector<mlir::OpFoldResult, 3> offsets = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(offsetOnLastAxis)};
    llvm::SmallVector<mlir::OpFoldResult, 3> sizes = {
        foldDim(batchDyn, batch), foldDim(seqDyn, seqLen),
        rewriter.getIndexAttr(qSize)};
    llvm::SmallVector<mlir::OpFoldResult, 3> strides(3,
                                                     rewriter.getIndexAttr(1));
    mlir::OperationState sliceState(
        loc, mlir::tensor::ExtractSliceOp::getOperationName());
    mlir::tensor::ExtractSliceOp::build(rewriter, sliceState, sliceType,
                                        qkvBiased, offsets, sizes, strides);
    return rewriter.create(sliceState)->getResult(0);
  };
  mlir::Value query = buildSlice(0);
  mlir::Value key = buildSlice(qSize);
  mlir::Value value = buildSlice(2 * qSize);

  // seqlens_k is a [B] i32 tensor. Bidirectional no-past attention ignores it
  // and treats every key as valid, so store the full sequence length. Causal
  // no-past attention uses ORT's prefill sentinel (-1): the runtime then sets
  // total_seq = S and past_len = 0. Storing S itself would be read as S+1 and
  // rejected because the present buffer only holds S keys.
  auto i32Ty = rewriter.getIntegerType(32);
  auto seqlensKType = mlir::RankedTensorType::get(
      {hiddenType.isDynamicDim(0) ? mlir::ShapedType::kDynamic : batch}, i32Ty);
  mlir::Value seqlensK;
  if (!noCausal) {
    if (batchDyn) {
      mlir::Value negOne = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(i32Ty, -1));
      seqlensK = mlir::tensor::GenerateOp::create(
                     rewriter, loc, seqlensKType, mlir::ValueRange{batchDyn},
                     [&](mlir::OpBuilder &b, mlir::Location bodyLoc,
                         mlir::ValueRange) {
                       mlir::tensor::YieldOp::create(b, bodyLoc, negOne);
                     })
                     .getResult();
    } else {
      auto attr = mlir::DenseElementsAttr::get(
          seqlensKType, rewriter.getIntegerAttr(i32Ty, -1));
      seqlensK = mlir::arith::ConstantOp::create(rewriter, loc, attr);
    }
  } else if (!batchDyn && !seqDyn) {
    llvm::SmallVector<llvm::APInt, 1> seqlensVals(
        static_cast<size_t>(batch), llvm::APInt(32, seqLen, /*isSigned=*/true));
    auto seqlensKAttr = mlir::DenseElementsAttr::get(
        seqlensKType, llvm::ArrayRef<llvm::APInt>(seqlensVals));
    seqlensK = mlir::arith::ConstantOp::create(rewriter, loc, seqlensKAttr);
  } else if (!batchDyn) {
    mlir::Value seqI32 =
        mlir::arith::IndexCastOp::create(rewriter, loc, i32Ty, seqDyn);
    llvm::SmallVector<mlir::Value> elems(static_cast<size_t>(batch), seqI32);
    seqlensK = mlir::tensor::FromElementsOp::create(rewriter, loc, seqlensKType,
                                                    elems);
  } else {
    mlir::Value seqI32;
    if (seqDyn)
      seqI32 = mlir::arith::IndexCastOp::create(rewriter, loc, i32Ty, seqDyn);
    else
      seqI32 = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(i32Ty, seqLen));
    seqlensK =
        mlir::tensor::GenerateOp::create(
            rewriter, loc, seqlensKType, mlir::ValueRange{batchDyn},
            [&](mlir::OpBuilder &b, mlir::Location bodyLoc, mlir::ValueRange) {
              mlir::tensor::YieldOp::create(b, bodyLoc, seqI32);
            })
            .getResult();
  }

  // The runtime derives the KV span from seqlens_k. total_seq_len is still a
  // required scalar and records the present-buffer capacity S.
  auto totalSeqLenType = mlir::RankedTensorType::get({}, i32Ty);
  mlir::Value totalSeqLen;
  if (!seqDyn) {
    auto totalSeqLenAttr = mlir::DenseElementsAttr::get(
        totalSeqLenType, llvm::APInt(32, seqLen, /*isSigned=*/true));
    totalSeqLen =
        mlir::arith::ConstantOp::create(rewriter, loc, totalSeqLenAttr);
  } else {
    mlir::Value seqI32 =
        mlir::arith::IndexCastOp::create(rewriter, loc, i32Ty, seqDyn);
    totalSeqLen = mlir::tensor::FromElementsOp::create(
        rewriter, loc, totalSeqLenType, mlir::ValueRange{seqI32});
  }

  const int64_t headSize = qSize / numHeads;
  auto presentType = mlir::RankedTensorType::get(
      {batch, numHeads, seqLen, headSize}, elemType);

  return buildHipGqaCall(op, rewriter, context, query, key, value,
                         /*pastKey=*/nullptr, /*pastValue=*/nullptr, seqlensK,
                         totalSeqLen, numHeads, scale, noCausal, outputType,
                         presentType, presentType, attentionBias);
}

} // namespace

void populateAttentionConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<AttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
