/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include "hip/Dialect/IR/HipShapeUtilsAttention.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <cmath>

namespace mlir {
namespace hip {
namespace {

/// onnx.Custom(com.microsoft.MultiHeadAttention) -> hip.{multi_head_attention,
/// gqa}.
///
/// Dispatch (in order):
///   1. 9-input MHA with past_key/past_value AND past_sequence_length, with
///      `past_present_share_buffer = 1` — this is Whisper decoder self-attn
///      AFTER the Task-9 ONNX surgery threads `past_sequence_length` into the
///      graph.  Lower to `hip.gqa(no_causal=false)` with `seqlens_k =
///      operands[8]` (runtime value).  Causal mask is kept (autoregressive).
///
///   2. 8-input MHA with EMPTY past_key/past_value slots — this is Whisper
///      decoder cross-attn (K/V come from encoder output, not a KV cache).
///      Lower to `hip.gqa(no_causal=true)` with `seqlens_k` as a constant
///      tensor [Skv,…,Skv] equal to the K's static sequence length.  Cross-
///      attention is bidirectional (Q can see all of K).
///
///   3. Plain separate-Q/K/V rank-3 fp16 MHA with one output lowers to
///      `hip.multi_head_attention`. Other forms are rejected before HIP IR is
///      emitted because the default runtime returns an error for them.
///
/// The constant `seqlens_k` / `total_sequence_length` for the cross-attn case
/// are emitted as `arith.constant` (not `onnx.Constant`): the greedy rewrite
/// driver runs with `ExistingOps` strictness, so newly-emitted onnx.* ops
/// would NOT be picked up by their lowering patterns in this pass.  See the
/// same comment in `AttentionConversion.cpp`.
struct MultiHeadAttentionToHip : public mlir::RewritePattern {
  MultiHeadAttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult MultiHeadAttentionToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "MultiHeadAttention")
    return rewriter.notifyMatchFailure(op,
                                       "not a MultiHeadAttention operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for MultiHeadAttention");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // MS MultiHeadAttention spec: 1-10 inputs.
  size_t numOps = op->getNumOperands();
  if (numOps < 1 || numOps > 10)
    return rewriter.notifyMatchFailure(
        op, "MultiHeadAttention expects 1-10 operands");
  size_t numResults = op->getNumResults();
  if (numResults < 1 || numResults > 4)
    return rewriter.notifyMatchFailure(
        op, "MultiHeadAttention expects 1-4 results");

  // Helper: get optional operand (check for NoneType / out-of-range).
  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps)
      return nullptr;
    mlir::Value val = op->getOperand(idx);
    if (mlir::isa<mlir::NoneType>(val.getType()))
      return nullptr;
    return val;
  };

  // === Extract Inputs (MS MultiHeadAttention spec order 1-10) ===

  // Input 1: query (required)
  mlir::Value query = op->getOperand(0);

  // Inputs 2-3: key / value (optional - packed QKV in query if both absent)
  mlir::Value key = getOptionalOperand(1);
  mlir::Value value = getOptionalOperand(2);

  // Input 4: bias (optional - input projection bias)
  mlir::Value bias = getOptionalOperand(3);

  // Input 5: key_padding_mask (optional)
  mlir::Value keyPaddingMask = getOptionalOperand(4);

  // Input 6: attention_bias (optional - added to Q*K')
  mlir::Value attentionBias = getOptionalOperand(5);

  // Inputs 7-8: past_key / past_value (optional - KV cache)
  mlir::Value pastKey = getOptionalOperand(6);
  mlir::Value pastValue = getOptionalOperand(7);

  // Input 9: past_sequence_length (optional - buffer sharing)
  mlir::Value pastSequenceLength = getOptionalOperand(8);

  // Input 10: cache_indirection (optional - beam search)
  mlir::Value cacheIndirection = getOptionalOperand(9);

  // === Extract Attributes ===

  auto numHeadsAttrOnnx = op->getAttrOfType<mlir::IntegerAttr>("num_heads");
  if (!numHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing num_heads attribute");
  auto numHeadsAttr =
      rewriter.getI64IntegerAttr(numHeadsAttrOnnx.getValue().getSExtValue());

  // Optional attribute helpers with defaults.
  auto getFloatAttr = [&](const char *name,
                          float defaultVal) -> mlir::FloatAttr {
    auto attr = op->getAttrOfType<mlir::FloatAttr>(name);
    return attr ? attr : rewriter.getF32FloatAttr(defaultVal);
  };

  auto getI64Attr = [&](const char *name,
                        int64_t defaultVal) -> mlir::IntegerAttr {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    // Convert ONNX signed integer to signless integer for HIP dialect.
    return attr ? rewriter.getI64IntegerAttr(attr.getValue().getSExtValue())
                : rewriter.getI64IntegerAttr(defaultVal);
  };

  // ONNX spec defaults: mask_filter_value = -10000.0, scale = 1/sqrt(head_size)
  // (we pass 0.0 as a runtime sentinel meaning "auto-compute at runtime",
  // matching the GQA convention).
  auto maskFilterValueAttr = getFloatAttr("mask_filter_value", -10000.0f);
  auto scaleAttr = getFloatAttr("scale", 0.0f);
  auto unidirectionalAttr = getI64Attr("unidirectional", 0);

  // === Whisper-style dispatch to hip.gqa ====================================
  //
  // Branch 2 (decoder self-attn after Task-9 surgery): 9-input form, both
  // past_* present, AND the past_sequence_length operand (slot 8) present.
  // Bias / mask / attention_bias / cache_indirection must all be absent —
  // those signals would require additional hip.gqa wiring that this
  // Whisper-targeted path doesn't model.
  //
  // The discriminator is the PRESENCE of the past_sequence_length operand, NOT
  // a past_present_share_buffer attribute. ORT's
  // com.microsoft.MultiHeadAttention schema rejects that attribute (it lives on
  // GroupQueryAttention/Attention), so the Task-9 surgery threads only the
  // input — and the input alone uniquely identifies the shared-buffer self-attn
  // form (cross-attn has empty slot 8; plain self-attn without surgery has no
  // slot 8 at all).
  const bool noExtraInputs =
      !bias && !keyPaddingMask && !attentionBias && !cacheIndirection;
  const bool hasFullSelfAttnTrio =
      key && value && pastKey && pastValue && pastSequenceLength;

  if (noExtraInputs && hasFullSelfAttnTrio) {
    // Decoder self-attn with past, post-surgery → hip.gqa(no_causal=false).
    //
    // seqlens_k: ONNX threads this as a [1]-i32 input named
    // `past_sequence_length` (1-D, batch=1 today).  hip.gqa expects a 1-D
    // [B]-i32 seqlens_k.  These match — we forward operand[8] directly.
    //
    // total_seq_len: the runtime infers `(past_seq_len + S)` itself from
    // seqlens_k + the query's S dim, so we don't need to compute it on the
    // MLIR side.  We pass `seqlens_k` again as a placeholder (any rank-0 or
    // rank-1 i32 tensor works since the kernel reads seqlens_k for the actual
    // KV length).  Actually hip.gqa REQUIRES a scalar total_seq_len, so we
    // emit a constant tensor of `present_key.shape[2]` — the static buffer
    // size from the IR (== `past_sequence_length_max + S`, baked in at compile
    // time).
    if (numResults != 3)
      return rewriter.notifyMatchFailure(
          op, "post-surgery self-attn GQA route requires output, present_key, "
              "and present_value");
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto presentKType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(1).getType());
    auto presentValueType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(2).getType());
    if (!outputType || !presentValueType)
      return rewriter.notifyMatchFailure(
          op, "post-surgery self-attn requires ranked output types");
    if (!presentKType || presentKType.getRank() != 4 ||
        presentKType.isDynamicDim(2))
      return rewriter.notifyMatchFailure(
          op, "post-surgery self-attn requires present_key with static dim 2");
    const int64_t totalLen = presentKType.getDimSize(2);

    auto i32Ty = rewriter.getIntegerType(32);
    auto totalSeqLenType = mlir::RankedTensorType::get({}, i32Ty);
    auto totalSeqLenAttr = mlir::DenseElementsAttr::get(
        totalSeqLenType, llvm::APInt(32, totalLen, /*isSigned=*/true));
    mlir::Value totalSeqLen =
        mlir::arith::ConstantOp::create(rewriter, loc, totalSeqLenAttr);

    return buildHipGqaCall(
        op, rewriter, context, query, key, value, pastKey, pastValue,
        /*seqlensK=*/pastSequenceLength, totalSeqLen,
        numHeadsAttr.getValue().getSExtValue(),
        scaleAttr.getValue().convertToFloat(), /*noCausal=*/false, outputType,
        presentKType, presentValueType);
  }

  // Branch 1 (decoder cross-attn): 8-input form, K/V supplied, past slots
  // EMPTY (no KV cache — K/V come directly from encoder output), no
  // bias/mask/attention_bias/cache_indirection.  The past_sequence_length
  // slot (input 9) is also empty for this form.
  //
  // CRITICAL: gate on `numOps >= 8` so plain 3-input MHA (basic self-attn
  // with just Q/K/V, no past slots even declared) keeps using the existing
  // hip.multi_head_attention path.  The Whisper cross-attn pattern is
  // recognisable specifically by the 8 (or more) operands with empty past
  // slots — a graph author that built the op with only 3 operands didn't
  // intend cross-attn semantics.
  if (numOps >= 8 && noExtraInputs && key && value && !pastKey && !pastValue &&
      !pastSequenceLength && !cacheIndirection) {
    if (numResults != 1)
      return rewriter.notifyMatchFailure(
          op, "cross-attn GQA route supports only the primary output");
    // Cross-attn: no_causal=true, seqlens_k = constant [Skv,…,Skv].
    //
    // Skv is K's sequence-length dim.  ONNX MHA's K can come in two layouts:
    //   * 3-D [B, Skv, hidden]            (separate K activation)
    //   * 4-D [B, num_heads, Skv, head]   (pre-split K cache — Whisper cross)
    // Detect by rank.
    auto keyType = mlir::dyn_cast<mlir::RankedTensorType>(key.getType());
    if (!keyType || !keyType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "cross-attn dispatch requires static-shape key");
    int64_t batch = keyType.getDimSize(0);
    int64_t skv;
    // We currently only support rank-4 BNSH cross-attention keys (the Whisper
    // path). Rank-3 keys would require synthesizing BNSH layout via a reshape;
    // no test exercises this path today, so reject loudly rather than ship
    // silent dead code.
    if (keyType.getRank() == 4)
      skv = keyType.getDimSize(2);
    else
      return rewriter.notifyMatchFailure(
          op, "cross-attn rank-3 key not yet supported; expected rank-4 BNSH");

    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto valueType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!outputType || !valueType)
      return rewriter.notifyMatchFailure(
          op, "cross-attn dispatch requires ranked output and value");

    auto i32Ty = rewriter.getIntegerType(32);
    auto seqlensKType = mlir::RankedTensorType::get({batch}, i32Ty);
    llvm::SmallVector<llvm::APInt, 1> seqlensVals(
        static_cast<size_t>(batch), llvm::APInt(32, skv, /*isSigned=*/true));
    auto seqlensKAttr = mlir::DenseElementsAttr::get(
        seqlensKType, llvm::ArrayRef<llvm::APInt>(seqlensVals));
    mlir::Value seqlensK =
        mlir::arith::ConstantOp::create(rewriter, loc, seqlensKAttr);

    auto totalSeqLenType = mlir::RankedTensorType::get({}, i32Ty);
    auto totalSeqLenAttr = mlir::DenseElementsAttr::get(
        totalSeqLenType, llvm::APInt(32, skv, /*isSigned=*/true));
    mlir::Value totalSeqLen =
        mlir::arith::ConstantOp::create(rewriter, loc, totalSeqLenAttr);

    // hip.gqa always emits present_key / present_value buffers.  Cross-attn
    // has no KV cache exposed at the ONNX boundary, so the present_* types
    // just mirror K/V (they're discarded after the call).  K is guaranteed to
    // be rank-4 BNSH here — the rank-3 case was rejected above.
    mlir::RankedTensorType presentKType = keyType;
    mlir::RankedTensorType presentVType = valueType;
    const int64_t numHeads = numHeadsAttr.getValue().getSExtValue();

    return buildHipGqaCall(
        op, rewriter, context, query, key, value,
        /*pastKey=*/nullptr, /*pastValue=*/nullptr, seqlensK, totalSeqLen,
        numHeads, scaleAttr.getValue().convertToFloat(),
        /*noCausal=*/true, outputType, presentKType, presentVType);
  }

  // === Default: runtime-supported hip.multi_head_attention subset ==========

  if (!key || !value)
    return op->emitError(
        "default MultiHeadAttention runtime requires separate key and value");
  if (bias || keyPaddingMask || attentionBias || pastKey || pastValue ||
      pastSequenceLength || cacheIndirection)
    return op->emitError(
        "default MultiHeadAttention runtime does not support bias, masks, "
        "past/cache inputs, or cache indirection");
  if (numResults != 1)
    return op->emitError(
        "default MultiHeadAttention runtime supports only the primary output; "
        "present_key, present_value, and qk are unsupported");

  auto queryType = mlir::dyn_cast<mlir::RankedTensorType>(query.getType());
  auto keyType = mlir::dyn_cast<mlir::RankedTensorType>(key.getType());
  auto valueType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!queryType || !keyType || !valueType || !outputType)
    return op->emitError(
        "default MultiHeadAttention runtime requires ranked Q/K/V and output");
  if (!queryType.getElementType().isF16() ||
      keyType.getElementType() != queryType.getElementType() ||
      valueType.getElementType() != queryType.getElementType() ||
      outputType.getElementType() != queryType.getElementType())
    return op->emitError(
        "default MultiHeadAttention runtime requires fp16 Q/K/V and output");
  if (unidirectionalAttr.getValue().getSExtValue() != 0 &&
      unidirectionalAttr.getValue().getSExtValue() != 1)
    return op->emitError("MultiHeadAttention unidirectional must be 0 or 1");
  if (maskFilterValueAttr.getValue().convertToFloat() != -10000.0f)
    return op->emitError(
        "default MultiHeadAttention runtime supports only the default "
        "mask_filter_value -10000");

  const int64_t numHeads = numHeadsAttr.getValue().getSExtValue();
  mlir::FailureOr<llvm::SmallVector<int64_t>> expectedShape =
      mlir::hip::inferMultiHeadAttentionOutputShape(
          queryType.getShape(), keyType.getShape(), valueType.getShape(),
          numHeads, [&]() { return op->emitError(); });
  if (mlir::failed(expectedShape))
    return mlir::failure();

  if (outputType.getRank() != static_cast<int64_t>(expectedShape->size()))
    return op->emitError(
        "MultiHeadAttention output rank contradicts Q/K/V shape contract");
  for (int64_t dim : llvm::seq<int64_t>(outputType.getRank())) {
    int64_t expected = (*expectedShape)[dim];
    int64_t actual = outputType.getDimSize(dim);
    if (mlir::ShapedType::isDynamic(expected)) {
      if (!mlir::ShapedType::isDynamic(actual))
        return op->emitError("MultiHeadAttention output dimension ")
               << dim << " must remain dynamic because query is dynamic";
    } else if (!mlir::ShapedType::isDynamic(actual) && actual != expected) {
      return op->emitError("MultiHeadAttention output dimension ")
             << dim << " must equal query extent " << expected;
    }
  }

  mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>> reifiedShape =
      mlir::hip::reifyMultiHeadAttentionOutputShape(
          rewriter, loc, query, key, value, numHeads,
          [&]() { return op->emitError(); });
  if (mlir::failed(reifiedShape))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> outputInit = createEmptyTensorFromReifiedShape(
      rewriter, loc, outputType, *reifiedShape);
  if (mlir::failed(outputInit))
    return op->emitError(
        "MultiHeadAttention output type contradicts Q/K/V shape contract");

  // === Build the new hip.multi_head_attention op ===

  mlir::SmallVector<mlir::Type> resultTypes = {outputType};

  mlir::SmallVector<mlir::Value> operands = {context, query, key, value,
                                             *outputInit};

  // Build named attributes.
  mlir::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("num_heads", numHeadsAttr));
  attrs.push_back(
      rewriter.getNamedAttr("mask_filter_value", maskFilterValueAttr));
  attrs.push_back(rewriter.getNamedAttr("scale", scaleAttr));
  attrs.push_back(rewriter.getNamedAttr("unidirectional", unidirectionalAttr));

  auto state = mlir::OperationState(loc, "hip.multi_head_attention");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes(resultTypes);

  // operand_segment_sizes for AttrSizedOperandSegments trait.
  // Segments match the order in HipOps.td:
  //   ctx(1), query(1), key(0|1), value(0|1), bias(0|1),
  //   key_padding_mask(0|1), attention_bias(0|1),
  //   past_key(0|1), past_value(0|1),
  //   past_sequence_length(0|1), cache_indirection(0|1),
  //   output(1), present_key(0|1), present_value(0|1), qk(0|1)
  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(1); // ctx
  segmentSizes.push_back(1); // query
  segmentSizes.push_back(1); // key
  segmentSizes.push_back(1); // value
  segmentSizes.push_back(0); // bias
  segmentSizes.push_back(0); // key_padding_mask
  segmentSizes.push_back(0); // attention_bias
  segmentSizes.push_back(0); // past_key
  segmentSizes.push_back(0); // past_value
  segmentSizes.push_back(0); // past_sequence_length
  segmentSizes.push_back(0); // cache_indirection
  segmentSizes.push_back(1); // output
  segmentSizes.push_back(0); // present_key
  segmentSizes.push_back(0); // present_value
  segmentSizes.push_back(0); // qk

  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));

  auto hipOp = rewriter.create(state);

  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateMultiHeadAttentionConversionPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *ctx) {
  patterns.add<MultiHeadAttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
