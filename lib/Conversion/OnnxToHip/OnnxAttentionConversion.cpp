/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Attention (ONNX opset 23/24) -> hip.gqa
///
/// Maps the standard ONNX scaled-dot-product attention op onto the existing
/// hip.gqa runtime.  RoPE is expected to be applied upstream (separate
/// onnx.RotaryEmbedding nodes).  External masks are threaded through the
/// hip.gqa attention_bias operand; the runtime skips the built-in causal mask
/// when either is_causal=0 or attention_bias is present.
///
/// Before (Gemma-style decoder layer, is_causal=0, external fp16 mask):
///   %y, %pk, %pv = onnx.Attention(%q, %k, %v, %mask, %past_k, %past_v)
///       {q_num_heads = 16, kv_num_heads = 8, is_causal = 0, scale = 1.0}
///       : (..., ..., ..., tensor<Bx1xSxTxf16>, ..., ...)
///       -> (..., tensor<BxGxTxDxf16>, tensor<BxGxTxDxf16>)
///
/// After:
///   %cur        = tensor.dim %q, %c1       // current KV tokens (== query seq)
///   %past       = tensor.dim %past_k, %c2  // valid past KV length (0 on prefill)
///   %tot        = arith.addi %past, %cur   // present seq = past + current
///   %seqlens_k  = tensor.from_elements (%tot - 1)  : tensor<1xi32>
///   %total_seq  = tensor.from_elements %tot        : tensor<i32>
///   %pk_init    = tensor.empty(%B, %tot)   // present sized to %tot, NOT %past
///   %y, %pk, %pv = hip.gqa(%ctx)
///       ins(%q, %k, %v, %past_k, %past_v, %seqlens_k, %total_seq, %mask)
///       outs(%y_init, %pk_init, %pv_init)
///       {num_heads = 16, kv_num_heads = 8, scale = 1.0, no_causal = true}
///
/// present KV is concat(past, current) along the seq axis, so its seq extent
/// is past_seq + current_seq.  A fresh prefill arrives with an EMPTY past
/// (past_seq == 0): sizing present from dim(past_key, 2) alone would collapse
/// it to a zero-length buffer, the output allocator would then hand the
/// runtime a null present_key/present_value, and wrap_group_query_attention
/// would reject the call and leave the attention output zero-filled.
struct OnnxAttentionToHip : public mlir::RewritePattern {
  OnnxAttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Attention", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult OnnxAttentionToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op->getLoc();

  size_t numOps = op->getNumOperands();
  if (numOps < 3 || numOps > 7)
    return rewriter.notifyMatchFailure(
        op, "onnx.Attention expects 3-7 operands (Q,K,V,[mask],[past_k],"
            "[past_v],[nonpad_kv_seqlen])");

  auto isLive = [](mlir::Value v) {
    return v && !mlir::isa<mlir::NoneType>(v.getType());
  };

  if (numOps == 7 && isLive(op->getOperand(6)))
    return rewriter.notifyMatchFailure(
        op, "nonpad_kv_seqlen (external-cache mode) not supported yet");

  mlir::Value query = op->getOperand(0);
  mlir::Value key = op->getOperand(1);
  mlir::Value value = op->getOperand(2);
  mlir::Value attnMask = numOps > 3 ? op->getOperand(3) : mlir::Value();
  mlir::Value pastKey = numOps > 4 ? op->getOperand(4) : mlir::Value();
  mlir::Value pastValue = numOps > 5 ? op->getOperand(5) : mlir::Value();

  if (attnMask && !isLive(attnMask))
    attnMask = nullptr;
  if (pastKey && !isLive(pastKey))
    pastKey = nullptr;
  if (pastValue && !isLive(pastValue))
    pastValue = nullptr;

  if ((pastKey == nullptr) != (pastValue == nullptr))
    return rewriter.notifyMatchFailure(
        op, "past_key and past_value must both be present or both absent");

  auto qNumHeadsAttr = op->getAttrOfType<mlir::IntegerAttr>("q_num_heads");
  auto kvNumHeadsAttr = op->getAttrOfType<mlir::IntegerAttr>("kv_num_heads");
  if (!qNumHeadsAttr || !kvNumHeadsAttr)
    return rewriter.notifyMatchFailure(
        op, "missing q_num_heads or kv_num_heads attribute");

  int64_t qNumHeads = qNumHeadsAttr.getValue().getSExtValue();
  int64_t kvNumHeads = kvNumHeadsAttr.getValue().getSExtValue();
  if (qNumHeads <= 0 || kvNumHeads <= 0)
    return rewriter.notifyMatchFailure(op, "head counts must be > 0");
  if (qNumHeads % kvNumHeads != 0)
    return rewriter.notifyMatchFailure(
        op, "q_num_heads must be divisible by kv_num_heads");

  auto getI64 = [&](const char *name, int64_t defaultVal) {
    auto a = op->getAttrOfType<mlir::IntegerAttr>(name);
    return a ? a.getValue().getSExtValue() : defaultVal;
  };
  auto getFloat = [&](const char *name, float defaultVal) {
    auto a = op->getAttrOfType<mlir::FloatAttr>(name);
    return a ? a.getValue().convertToFloat() : defaultVal;
  };

  int64_t isCausal = getI64("is_causal", 0);
  float scale = getFloat("scale", 0.0f);
  float softcap = getFloat("softcap", 0.0f);

  // is_causal=0: mask (if any) carries causal+padding; skip built-in causal.
  // is_causal=1: built-in causal mask; optional attn_mask is additive only.
  bool noCausal = (isCausal == 0);

  if (isCausal == 0 && !attnMask)
    return rewriter.notifyMatchFailure(
        op, "is_causal=0 requires an attn_mask operand");

  size_t numResults = op->getNumResults();
  if (numResults < 1 || numResults > 4)
    return rewriter.notifyMatchFailure(
        op, "onnx.Attention expects 1-4 results (Y,[present_k],[present_v],"
            "[qk])");
  if (numResults > 3)
    return rewriter.notifyMatchFailure(op, "qk_matmul_output not supported yet");

  auto queryType = mlir::dyn_cast<mlir::RankedTensorType>(query.getType());
  if (!queryType || queryType.getRank() != 3)
    return rewriter.notifyMatchFailure(
        op, "query must be a rank-3 tensor [B, S, q_hidden]");

  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::RankedTensorType presentKeyType;
  mlir::RankedTensorType presentValueType;
  if (numResults >= 3) {
    presentKeyType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
    presentValueType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(2).getType());
  } else {
    return rewriter.notifyMatchFailure(
        op, "present_key/present_value outputs required for hip.gqa lowering");
  }

  if (presentKeyType.getRank() != 4 || presentValueType.getRank() != 4)
    return rewriter.notifyMatchFailure(
        op, "present_key/value must be rank-4 BNSH tensors");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  auto i32Ty = rewriter.getIntegerType(32);

  // present KV = concat(past, current) along the seq axis, so the present seq
  // extent is past_seq + current_seq. The current KV token count equals the
  // query sequence length for self-attention (Q/K/V share the seq axis); query
  // is validated rank-3 [B, S, hidden] above, so take S from query dim 1. The
  // valid past length is dim(past_key, 2) (rank-4 BNSH), or 0 when there is no
  // past operand (a fresh prefill carries an EMPTY past, past_seq == 0).
  mlir::Value curSeqIdx =
      mlir::tensor::DimOp::create(rewriter, loc, query, 1).getResult();
  mlir::Value pastSeqIdx =
      pastKey
          ? mlir::tensor::DimOp::create(rewriter, loc, pastKey, 2).getResult()
          : mlir::arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();
  mlir::Value totalKvIdx =
      mlir::arith::AddIOp::create(rewriter, loc, pastSeqIdx, curSeqIdx)
          .getResult();

  // seqlens_k[b] = total_seq - 1. The runtime uses the ORT GQA convention
  // total_seq = seqlens_k + 1 and derives past_len = total_seq - sq, which for
  // self-attention (current_seq == sq) is exactly past_seq. (Was dim(past_key,
  // 2) = past_seq, which under +1 gives total_seq = past_seq + 1 — correct only
  // for single-token decode, and 1 for an empty-past prefill.)
  mlir::Value oneIdx =
      mlir::arith::ConstantIndexOp::create(rewriter, loc, 1).getResult();
  mlir::Value totalKvM1Idx =
      mlir::arith::SubIOp::create(rewriter, loc, totalKvIdx, oneIdx).getResult();
  mlir::Value seqlensKI32 =
      mlir::arith::IndexCastOp::create(rewriter, loc, i32Ty, totalKvM1Idx);
  auto seqlensKType = mlir::RankedTensorType::get({1}, i32Ty);
  mlir::Value seqlensK = mlir::tensor::FromElementsOp::create(
      rewriter, loc, seqlensKType, seqlensKI32);

  // total_seq_len scalar = present KV buffer capacity = past_seq + current_seq.
  // (The runtime re-derives total_seq from seqlens_k today, but keep this
  // consistent with the present buffer for any future consumer.)
  mlir::Value totalSeqLen;
  if (!presentKeyType.isDynamicDim(2)) {
    auto totalSeqLenType = mlir::RankedTensorType::get({}, i32Ty);
    auto totalSeqLenAttr = mlir::DenseElementsAttr::get(
        totalSeqLenType,
        llvm::APInt(32, presentKeyType.getDimSize(2), /*isSigned=*/true));
    totalSeqLen =
        mlir::arith::ConstantOp::create(rewriter, loc, totalSeqLenAttr);
  } else {
    mlir::Value totalKvI32 =
        mlir::arith::IndexCastOp::create(rewriter, loc, i32Ty, totalKvIdx);
    auto totalSeqLenType = mlir::RankedTensorType::get({}, i32Ty);
    totalSeqLen = mlir::tensor::FromElementsOp::create(
        rewriter, loc, totalSeqLenType, totalKvI32);
  }

  // Build the present_key/present_value init buffers with seq dim = total_seq
  // (past + current). present is rank-4 BNSH [batch, kv_heads, seq, head_dim];
  // batch (dim 0) and seq (dim 2) are the only dynamic dims in practice —
  // kv_heads/head_dim are architecture constants. Batch is taken from query
  // dim 0; the seq extent is the past+current total computed above.
  auto buildPresentInit =
      [&](mlir::RankedTensorType t) -> mlir::FailureOr<mlir::Value> {
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t dimIdx : llvm::seq<int64_t>(t.getRank())) {
      if (!t.isDynamicDim(dimIdx))
        continue;
      if (dimIdx == 2)
        dynSizes.push_back(totalKvIdx);
      else if (dimIdx == 0)
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, query, 0).getResult());
      else if (pastKey)
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, pastKey, dimIdx)
                .getResult());
      else
        return mlir::failure();
    }
    return mlir::Value(mlir::tensor::EmptyOp::create(
        rewriter, loc, t.getShape(), t.getElementType(), dynSizes));
  };

  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, query);
  mlir::FailureOr<mlir::Value> presentKeyInitOr =
      buildPresentInit(presentKeyType);
  mlir::FailureOr<mlir::Value> presentValueInitOr =
      buildPresentInit(presentValueType);
  if (mlir::failed(presentKeyInitOr) || mlir::failed(presentValueInitOr))
    return rewriter.notifyMatchFailure(
        op, "present_key/present_value has an unsupported dynamic dim "
            "(only batch and seq may be dynamic)");
  mlir::Value presentKeyInit = *presentKeyInitOr;
  mlir::Value presentValueInit = *presentValueInitOr;

  llvm::SmallVector<mlir::Type> resultTypes = {outputType, presentKeyType,
                                               presentValueType};

  llvm::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(query);
  operands.push_back(key);
  operands.push_back(value);
  if (pastKey)
    operands.push_back(pastKey);
  if (pastValue)
    operands.push_back(pastValue);
  operands.push_back(seqlensK);
  operands.push_back(totalSeqLen);
  if (attnMask)
    operands.push_back(attnMask);
  operands.push_back(outputInit);
  operands.push_back(presentKeyInit);
  operands.push_back(presentValueInit);

  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(1); // ctx
  segmentSizes.push_back(1); // query
  segmentSizes.push_back(1); // key
  segmentSizes.push_back(1); // value
  segmentSizes.push_back(pastKey ? 1 : 0);
  segmentSizes.push_back(pastValue ? 1 : 0);
  segmentSizes.push_back(1); // seqlens_k
  segmentSizes.push_back(1); // total_seq_len
  segmentSizes.push_back(0); // cos_cache
  segmentSizes.push_back(0); // sin_cache
  segmentSizes.push_back(0); // position_ids
  segmentSizes.push_back(attnMask ? 1 : 0); // attention_bias
  segmentSizes.push_back(0);                // head_sink
  segmentSizes.push_back(0);                // k_scale
  segmentSizes.push_back(0);                // v_scale
  segmentSizes.push_back(1);                // output
  segmentSizes.push_back(1);                // present_key
  segmentSizes.push_back(1);                // present_value
  segmentSizes.push_back(0);                // output_qk

  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "num_heads", rewriter.getI64IntegerAttr(qNumHeads)));
  attrs.push_back(rewriter.getNamedAttr(
      "kv_num_heads", rewriter.getI64IntegerAttr(kvNumHeads)));
  attrs.push_back(
      rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)));
  attrs.push_back(
      rewriter.getNamedAttr("softcap", rewriter.getF32FloatAttr(softcap)));
  attrs.push_back(
      rewriter.getNamedAttr("no_causal", rewriter.getBoolAttr(noCausal)));

  mlir::OperationState gqaState(loc, "hip.gqa");
  gqaState.addOperands(operands);
  gqaState.addTypes(resultTypes);
  gqaState.addAttributes(attrs);
  gqaState.addAttribute("operand_segment_sizes",
                        rewriter.getDenseI32ArrayAttr(segmentSizes));
  mlir::Operation *gqaOp = rewriter.create(gqaState);

  rewriter.replaceOp(op, gqaOp->getResults());
  return mlir::success();
}

} // namespace

void populateOnnxAttentionConversionPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx) {
  patterns.add<OnnxAttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
