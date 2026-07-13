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
///   %past_len   = tensor.dim %past_k, %c2
///   %seqlens_k  = tensor.from_elements %past_len_cast : tensor<1xi32>
///   %total_seq  = tensor.from_elements %present_len  : tensor<i32>
///   %y, %pk, %pv = hip.gqa(%ctx)
///       ins(%q, %k, %v, %past_k, %past_v, %seqlens_k, %total_seq, %mask)
///       outs(...)
///       {num_heads = 16, kv_num_heads = 8, scale = 1.0, no_causal = true}
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
  mlir::Value c2 =
      mlir::arith::ConstantIndexOp::create(rewriter, loc, 2).getResult();

  // seqlens_k[b] = valid past length = dim(past_key, seq).
  mlir::Value seqlensK;
  if (pastKey) {
    mlir::Value pastSeqIdx =
        mlir::tensor::DimOp::create(rewriter, loc, pastKey, c2).getResult();
    mlir::Value pastSeqI32 = mlir::arith::IndexCastOp::create(
        rewriter, loc, i32Ty, pastSeqIdx);
    auto seqlensKType = mlir::RankedTensorType::get({1}, i32Ty);
    seqlensK = mlir::tensor::FromElementsOp::create(rewriter, loc, seqlensKType,
                                                   pastSeqI32);
  } else {
    // Prefill with empty past: ORT sentinel -1 (fresh prefill, past_len=0).
    auto seqlensKType = mlir::RankedTensorType::get({1}, i32Ty);
    auto sentinel = mlir::DenseElementsAttr::get(
        seqlensKType, llvm::APInt(32, -1, /*isSigned=*/true));
    seqlensK = mlir::arith::ConstantOp::create(rewriter, loc, sentinel);
  }

  // total_seq_len scalar = present_key buffer seq capacity (dim 2).
  mlir::Value totalSeqLen;
  if (!presentKeyType.isDynamicDim(2)) {
    auto totalSeqLenType = mlir::RankedTensorType::get({}, i32Ty);
    auto totalSeqLenAttr = mlir::DenseElementsAttr::get(
        totalSeqLenType,
        llvm::APInt(32, presentKeyType.getDimSize(2), /*isSigned=*/true));
    totalSeqLen =
        mlir::arith::ConstantOp::create(rewriter, loc, totalSeqLenAttr);
  } else if (pastKey) {
    mlir::Value presentSeqIdx =
        mlir::tensor::DimOp::create(rewriter, loc, pastKey, c2).getResult();
    mlir::Value presentSeqI32 = mlir::arith::IndexCastOp::create(
        rewriter, loc, i32Ty, presentSeqIdx);
    auto totalSeqLenType = mlir::RankedTensorType::get({}, i32Ty);
    totalSeqLen = mlir::tensor::FromElementsOp::create(
        rewriter, loc, totalSeqLenType, presentSeqI32);
  } else {
    return rewriter.notifyMatchFailure(
        op, "dynamic present_key seq dim requires past_key operand");
  }

  mlir::Value outputInit =
      createEmptyTensor(rewriter, loc, outputType, query);
  mlir::Value presentKeyInit = createEmptyTensor(
      rewriter, loc, presentKeyType, pastKey ? pastKey : query);
  mlir::Value presentValueInit = createEmptyTensor(
      rewriter, loc, presentValueType, pastValue ? pastValue : query);

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
