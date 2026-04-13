/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.scaled_dot_product_attention -> hip.gqa
///
/// PyTorch SDPA signature:
///   %out = "torch.aten.scaled_dot_product_attention"(
///       %query, %key, %value, %attn_mask, %dropout_p, %is_causal, %scale)
///
/// Q/K/V are in BHSD format: [batch, num_heads, seq_len, head_dim]
/// hip.gqa expects Q in BSD format: [batch, seq_len, num_heads * head_dim]
/// and K/V in BSD format: [batch, seq_len, kv_num_heads * head_dim]
///
/// This conversion:
/// 1. Extracts num_heads, kv_num_heads, head_dim from Q/K shapes
/// 2. Reshapes Q from [B,H,S,D] → [B,S,H*D] (transpose + reshape)
/// 3. Reshapes K from [B,Hkv,S,D] → [B,S,Hkv*D]
/// 4. Reshapes V from [B,Hkv,S,D] → [B,S,Hkv*D]
/// 5. Creates seqlens_k and total_seq_len auxiliary tensors
/// 6. Emits hip.gqa with no KV cache, no RoPE
struct TorchSdpaToGqa : public mlir::RewritePattern {
  TorchSdpaToGqa(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.scaled_dot_product_attention",
                       /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();

    // Operands: query, key, value, [attn_mask], [dropout_p], [is_causal],
    // [scale]
    mlir::Value query = op->getOperand(0); // [B, H, S, D]
    mlir::Value key = op->getOperand(1);   // [B, Hkv, S, D]
    mlir::Value value = op->getOperand(2); // [B, Hkv, S, D]

    auto queryType = mlir::dyn_cast<mlir::RankedTensorType>(query.getType());
    auto keyType = mlir::dyn_cast<mlir::RankedTensorType>(key.getType());
    if (!queryType || !keyType)
      return rewriter.notifyMatchFailure(op, "Q/K must be ranked tensors");

    int64_t qRank = queryType.getRank();
    int64_t kRank = keyType.getRank();

    if (qRank != 3 && qRank != 4)
      return rewriter.notifyMatchFailure(op, "Q must be rank-3 [B,S,H*D] "
                                             "or rank-4 [B,H,S,D]");
    if (kRank != qRank)
      return rewriter.notifyMatchFailure(op, "K rank must match Q rank");

    auto elemType = queryType.getElementType();
    int64_t batch, numHeads, seqLen, headDim, kvNumHeads;
    int64_t qHidden, kvHidden;
    bool needsTranspose = (qRank == 4);

    if (qRank == 4) {
      // BHSD format: [batch, num_heads, seq_len, head_dim]
      batch = queryType.getDimSize(0);
      numHeads = queryType.getDimSize(1);
      seqLen = queryType.getDimSize(2);
      headDim = queryType.getDimSize(3);
      kvNumHeads = keyType.getDimSize(1);
    } else {
      // BSD format: [batch, seq_len, hidden_size]
      batch = queryType.getDimSize(0);
      seqLen = queryType.getDimSize(1);
      qHidden = queryType.getDimSize(2);
      kvHidden = keyType.getDimSize(2);
      // Infer num_heads from hidden/kv_hidden ratio
      // For same-head attention: num_heads = kv_num_heads, head_dim =
      // hidden/heads We assume qHidden and kvHidden are available Default:
      // assume head_dim divides into hidden evenly For GQA: num_heads may
      // differ from kv_num_heads Without additional info, assume kv_num_heads =
      // num_heads for BSD input
      headDim = 0;
      numHeads = 0;
      kvNumHeads = 0;
      // We'll set these from shape below
    }

    if (needsTranspose) {
      if (batch == mlir::ShapedType::kDynamic ||
          numHeads == mlir::ShapedType::kDynamic ||
          seqLen == mlir::ShapedType::kDynamic ||
          headDim == mlir::ShapedType::kDynamic ||
          kvNumHeads == mlir::ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(
            op, "dynamic shapes not yet supported in SDPA->GQA conversion");
      qHidden = numHeads * headDim;
      kvHidden = kvNumHeads * headDim;
    } else {
      // BSD: infer head counts
      if (batch == mlir::ShapedType::kDynamic ||
          seqLen == mlir::ShapedType::kDynamic ||
          qHidden == mlir::ShapedType::kDynamic ||
          kvHidden == mlir::ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(
            op, "dynamic shapes not yet supported in SDPA->GQA conversion");
      // Infer: for same-head attention, qHidden == kvHidden, heads = gcd
      // For now assume same heads: num_heads = kv_num_heads
      // head_dim = qHidden / num_heads, pick largest head_dim that divides both
      // Common head dims: 32, 64, 128
      for (int64_t hd : {128, 64, 32}) {
        if (qHidden % hd == 0 && kvHidden % hd == 0) {
          headDim = hd;
          numHeads = qHidden / hd;
          kvNumHeads = kvHidden / hd;
          break;
        }
      }
      if (headDim == 0)
        return rewriter.notifyMatchFailure(
            op, "cannot infer head_dim from BSD shapes");
    }

    mlir::Value qBsd, kBsd, vBsd;
    auto qBsdType =
        mlir::RankedTensorType::get({batch, seqLen, qHidden}, elemType);
    auto kBsdType =
        mlir::RankedTensorType::get({batch, seqLen, kvHidden}, elemType);
    auto vBsdType = kBsdType;

    if (needsTranspose) {
      // BHSD → BSD: transpose dims 1,2 then collapse last two
      mlir::Value dim1Val =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      mlir::Value dim2Val =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 2);
      llvm::SmallVector<mlir::ReassociationIndices> reassoc = {
          {0}, {1}, {2, 3}};

      auto qTransType = mlir::RankedTensorType::get(
          {batch, seqLen, numHeads, headDim}, elemType);
      auto qTrans = mlir::hip::TransposeOp::create(
          rewriter, loc, qTransType, context, dim1Val, dim2Val, query,
          createEmptyTensorForTorch(rewriter, loc, qTransType, query));
      qBsd = mlir::tensor::CollapseShapeOp::create(
          rewriter, loc, qBsdType, qTrans->getResult(0), reassoc);

      auto kTransType = mlir::RankedTensorType::get(
          {batch, seqLen, kvNumHeads, headDim}, elemType);
      auto kTrans = mlir::hip::TransposeOp::create(
          rewriter, loc, kTransType, context, dim1Val, dim2Val, key,
          createEmptyTensorForTorch(rewriter, loc, kTransType, key));
      kBsd = mlir::tensor::CollapseShapeOp::create(
          rewriter, loc, kBsdType, kTrans->getResult(0), reassoc);

      auto vTrans = mlir::hip::TransposeOp::create(
          rewriter, loc, kTransType, context, dim1Val, dim2Val, value,
          createEmptyTensorForTorch(rewriter, loc, kTransType, value));
      vBsd = mlir::tensor::CollapseShapeOp::create(
          rewriter, loc, vBsdType, vTrans->getResult(0), reassoc);
    } else {
      // Already BSD format
      qBsd = query;
      kBsd = key;
      vBsd = value;
    }

    // Create seqlens_k: [batch] filled with seqLen-1 (0-indexed last valid pos)
    // For prefill (no KV cache), seqlens_k = seqLen - 1 for each batch
    auto seqlensType =
        mlir::RankedTensorType::get({batch, 1}, rewriter.getI32Type());
    auto totalSeqType = mlir::RankedTensorType::get({}, rewriter.getI32Type());

    // Create constant seqlens_k and total_seq_len via arith.constant
    auto seqlensVal = rewriter.getI32IntegerAttr(seqLen - 1);
    auto totalSeqVal = rewriter.getI32IntegerAttr(seqLen);

    auto seqlensAttr = mlir::DenseElementsAttr::get(seqlensType, seqlensVal);
    auto totalSeqAttr = mlir::DenseElementsAttr::get(totalSeqType, totalSeqVal);

    auto seqlensK = mlir::arith::ConstantOp::create(rewriter, loc, seqlensType,
                                                    seqlensAttr);
    auto totalSeqLen = mlir::arith::ConstantOp::create(
        rewriter, loc, totalSeqType, totalSeqAttr);

    // Output type: [B, S, H*D] (same as Q in BSD format)
    auto outputType = qBsdType;
    mlir::Value outputInit =
        createEmptyTensorForTorch(rewriter, loc, outputType, qBsd);

    // Present key/value: [B, Hkv, S, D] (BNSH format for cache output)
    auto presentKeyType = mlir::RankedTensorType::get(
        {batch, kvNumHeads, seqLen, headDim}, elemType);
    auto presentValueType = presentKeyType;
    mlir::Value presentKeyInit =
        createEmptyTensorForTorch(rewriter, loc, presentKeyType, key);
    mlir::Value presentValueInit =
        createEmptyTensorForTorch(rewriter, loc, presentValueType, value);

    // Compute scale
    float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

    // Build operands
    mlir::SmallVector<mlir::Value> operands;
    operands.push_back(context);
    operands.push_back(qBsd); // query
    operands.push_back(kBsd); // key
    operands.push_back(vBsd); // value
    // no past_key, no past_value
    operands.push_back(seqlensK);    // seqlens_k
    operands.push_back(totalSeqLen); // total_seq_len
    // no cos_cache, sin_cache, position_ids, attention_bias, head_sink,
    // k_scale, v_scale
    operands.push_back(outputInit);
    operands.push_back(presentKeyInit);
    operands.push_back(presentValueInit);
    // no output_qk

    // Build attributes
    mlir::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "num_heads", rewriter.getI64IntegerAttr(numHeads)));
    attrs.push_back(rewriter.getNamedAttr(
        "kv_num_heads", rewriter.getI64IntegerAttr(kvNumHeads)));
    attrs.push_back(
        rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)));
    attrs.push_back(
        rewriter.getNamedAttr("do_rotary", rewriter.getI64IntegerAttr(0)));
    attrs.push_back(rewriter.getNamedAttr("rotary_interleaved",
                                          rewriter.getI64IntegerAttr(0)));
    attrs.push_back(
        rewriter.getNamedAttr("softcap", rewriter.getF32FloatAttr(0.0f)));
    attrs.push_back(rewriter.getNamedAttr("local_window_size",
                                          rewriter.getI64IntegerAttr(-1)));
    attrs.push_back(
        rewriter.getNamedAttr("smooth_softmax", rewriter.getI64IntegerAttr(0)));
    attrs.push_back(
        rewriter.getNamedAttr("qk_output", rewriter.getI64IntegerAttr(0)));
    attrs.push_back(
        rewriter.getNamedAttr("k_quant_type", rewriter.getStringAttr("NONE")));
    attrs.push_back(
        rewriter.getNamedAttr("v_quant_type", rewriter.getStringAttr("NONE")));
    attrs.push_back(rewriter.getNamedAttr("kv_cache_bit_width",
                                          rewriter.getI64IntegerAttr(8)));

    // operand_segment_sizes:
    // ctx(1) query(1) key(1) value(1) past_key(0) past_value(0)
    // seqlens_k(1) total_seq_len(1) cos_cache(0) sin_cache(0)
    // position_ids(0) attention_bias(0) head_sink(0) k_scale(0) v_scale(0)
    // output(1) present_key(1) present_value(1) output_qk(0)
    llvm::SmallVector<int32_t> segSizes = {1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
                                           0, 0, 0, 0, 0, 1, 1, 1, 0};

    mlir::SmallVector<mlir::Type> resultTypes = {outputType, presentKeyType,
                                                 presentValueType};

    auto state = mlir::OperationState(loc, "hip.gqa");
    state.addOperands(operands);
    state.addAttributes(attrs);
    state.addTypes(resultTypes);
    state.addAttribute("operand_segment_sizes",
                       rewriter.getDenseI32ArrayAttr(segSizes));

    auto hipOp = rewriter.create(state);

    auto outputBsd = hipOp->getResult(0);

    if (needsTranspose) {
      // Reshape GQA output from BSD [B,S,H*D] back to BHSD [B,H,S,D]
      auto expandedType = mlir::RankedTensorType::get(
          {batch, seqLen, numHeads, headDim}, elemType);
      llvm::SmallVector<mlir::ReassociationIndices> expandReassoc = {
          {0}, {1}, {2, 3}};
      auto expanded = mlir::tensor::ExpandShapeOp::create(
          rewriter, loc, expandedType, outputBsd, expandReassoc);

      // Transpose [B,S,H,D] → [B,H,S,D]
      mlir::Value dim1Val =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      mlir::Value dim2Val =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 2);
      auto resultType =
          mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
      mlir::Value outInit =
          createEmptyTensorForTorch(rewriter, loc, resultType, expanded);
      auto transposed =
          mlir::hip::TransposeOp::create(rewriter, loc, resultType, context,
                                         dim1Val, dim2Val, expanded, outInit);
      rewriter.replaceOp(op, transposed->getResult(0));
    } else {
      // BSD input → BSD output, no transpose needed
      rewriter.replaceOp(op, outputBsd);
    }
    return mlir::success();
  }
};

} // namespace

void populateTorchGqaConversionPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::MLIRContext *ctx) {
  patterns.add<TorchSdpaToGqa>(ctx);
}

} // namespace hip
} // namespace mlir
