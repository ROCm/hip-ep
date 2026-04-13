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
    if (!queryType || !keyType || queryType.getRank() != 4 ||
        keyType.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "Q/K must be rank-4 [B,H,S,D]");

    int64_t batch = queryType.getDimSize(0);
    int64_t numHeads = queryType.getDimSize(1);
    int64_t seqLen = queryType.getDimSize(2);
    int64_t headDim = queryType.getDimSize(3);
    int64_t kvNumHeads = keyType.getDimSize(1);

    if (batch == mlir::ShapedType::kDynamic ||
        numHeads == mlir::ShapedType::kDynamic ||
        seqLen == mlir::ShapedType::kDynamic ||
        headDim == mlir::ShapedType::kDynamic ||
        kvNumHeads == mlir::ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(op, "dynamic shapes not yet supported "
                                              "in SDPA->GQA conversion");

    auto elemType = queryType.getElementType();
    int64_t qHidden = numHeads * headDim;
    int64_t kvHidden = kvNumHeads * headDim;

    // Helper: create dim index constants
    mlir::Value dim1Val =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    mlir::Value dim2Val =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, 2);

    // Reshape Q: [B,H,S,D] → [B,S,H,D] → [B,S,H*D]
    // Transpose dims 1,2: [B,H,S,D] → [B,S,H,D]
    auto qTransposedType =
        mlir::RankedTensorType::get({batch, seqLen, numHeads, headDim}, elemType);
    mlir::Value qInit =
        createEmptyTensorForTorch(rewriter, loc, qTransposedType, query);
    auto qTransposed = mlir::hip::TransposeOp::create(
        rewriter, loc, qTransposedType, context, dim1Val, dim2Val, query,
        qInit);

    // Collapse [B,S,H,D] → [B,S,H*D]
    auto qBsdType =
        mlir::RankedTensorType::get({batch, seqLen, qHidden}, elemType);
    llvm::SmallVector<mlir::ReassociationIndices> reassoc3to2 = {
        {0}, {1}, {2, 3}};
    auto qBsd = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, qBsdType, qTransposed->getResult(0), reassoc3to2);

    // Reshape K: [B,Hkv,S,D] → [B,S,Hkv,D] → [B,S,Hkv*D]
    auto kTransposedType = mlir::RankedTensorType::get(
        {batch, seqLen, kvNumHeads, headDim}, elemType);
    mlir::Value kInit =
        createEmptyTensorForTorch(rewriter, loc, kTransposedType, key);
    auto kTransposed = mlir::hip::TransposeOp::create(
        rewriter, loc, kTransposedType, context, dim1Val, dim2Val, key, kInit);

    auto kBsdType =
        mlir::RankedTensorType::get({batch, seqLen, kvHidden}, elemType);
    auto kBsd = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, kBsdType, kTransposed->getResult(0), reassoc3to2);

    // Reshape V: same as K
    mlir::Value vInit =
        createEmptyTensorForTorch(rewriter, loc, kTransposedType, value);
    auto vTransposed = mlir::hip::TransposeOp::create(
        rewriter, loc, kTransposedType, context, dim1Val, dim2Val, value,
        vInit);

    auto vBsdType = kBsdType;
    auto vBsd = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, vBsdType, vTransposed->getResult(0), reassoc3to2);

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

    auto seqlensK =
        mlir::arith::ConstantOp::create(rewriter, loc, seqlensType, seqlensAttr);
    auto totalSeqLen =
        mlir::arith::ConstantOp::create(rewriter, loc, totalSeqType, totalSeqAttr);

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
    operands.push_back(qBsd);       // query
    operands.push_back(kBsd);       // key
    operands.push_back(vBsd);       // value
    // no past_key, no past_value
    operands.push_back(seqlensK);   // seqlens_k
    operands.push_back(totalSeqLen); // total_seq_len
    // no cos_cache, sin_cache, position_ids, attention_bias, head_sink,
    // k_scale, v_scale
    operands.push_back(outputInit);
    operands.push_back(presentKeyInit);
    operands.push_back(presentValueInit);
    // no output_qk

    // Build attributes
    mlir::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("num_heads", rewriter.getI64IntegerAttr(numHeads)));
    attrs.push_back(rewriter.getNamedAttr("kv_num_heads",
                                          rewriter.getI64IntegerAttr(kvNumHeads)));
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
    attrs.push_back(rewriter.getNamedAttr("smooth_softmax",
                                          rewriter.getI64IntegerAttr(0)));
    attrs.push_back(
        rewriter.getNamedAttr("qk_output", rewriter.getI64IntegerAttr(0)));
    attrs.push_back(rewriter.getNamedAttr("k_quant_type",
                                          rewriter.getStringAttr("NONE")));
    attrs.push_back(rewriter.getNamedAttr("v_quant_type",
                                          rewriter.getStringAttr("NONE")));
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

    // The original SDPA returns a single tensor in BHSD format.
    // We need to reshape the GQA output from BSD [B,S,H*D] back to
    // BHSD [B,H,S,D].
    auto outputBsd = hipOp->getResult(0);

    // Expand [B,S,H*D] → [B,S,H,D]
    auto expandedType = mlir::RankedTensorType::get(
        {batch, seqLen, numHeads, headDim}, elemType);
    llvm::SmallVector<mlir::ReassociationIndices> expandReassoc = {
        {0}, {1}, {2, 3}};
    auto expanded = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, expandedType, outputBsd, expandReassoc);

    // Transpose [B,S,H,D] → [B,H,S,D]
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value outInit =
        createEmptyTensorForTorch(rewriter, loc, resultType, expanded);
    auto transposed = mlir::hip::TransposeOp::create(
        rewriter, loc, resultType, context, dim1Val, dim2Val, expanded, outInit);

    rewriter.replaceOp(op, transposed->getResult(0));
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
