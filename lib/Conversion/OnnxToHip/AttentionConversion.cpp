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

/// onnx.Custom(com.microsoft.Attention) -> hip.gqa (bidirectional self-attn).
///
/// Whisper's encoder uses the legacy com.microsoft.Attention op with a single
/// fused QKV projection weight + bias.  We rewrite each such node to:
///
///   1. One QKV projection over the fused weight (no transpose):
///        qkv_proj  = onnx.MatMul(hidden, qkv_w)         : [B,S,H] @ [H,3H]
///                                                            -> [B,S,3H]
///        qkv_biased = onnx.Add(qkv_proj, qkv_b)         : broadcast on last
///        dim
///      The ORT com.microsoft.Attention spec stores the fused weight as
///      [input_hidden, q_hidden+k_hidden+v_hidden] = [H, 3H] (what Whisper
///      actually exports — `qkv_proj.weight` is [1280, 3840]).  This is the
///      canonical layout, so the matmul consumes qkv_w directly — no transpose.
///
///   2. Three tensor.extract_slice ops splitting the last axis of qkv_biased
///      into Q / K / V (`qkv_hidden_sizes`-driven offsets).
///
///   3. One hip.gqa with `no_causal = true` (encoder is bidirectional),
///      `num_heads == kv_num_heads` (HPG=1, multi-head not group-query),
///      and a compile-time `seqlens_k` / `total_sequence_length` constant
///      equal to the static query sequence length S.  No past_key/past_value
///      (encoder self-attn has no KV cache); the present_* outputs are
///      DPS-style empty buffers that just satisfy the hip.gqa signature.
///
/// We keep the QKV projection fused (Option B) rather than splitting the
/// weight into three independent MatMul/Add chains (Option A): the fused
/// layout matches what the original ONNX graph already has and produces a
/// single large GEMM (cleaner lowering, fewer hipBLASLt launches).
struct AttentionToHip : public mlir::RewritePattern {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttentionToHip)
  AttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

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

  // === Operands (MS Attention spec) ===
  // Required: input(hidden), weights(qkv_w), bias(qkv_b).
  // Optional (5..8): mask_index, past, attention_bias, past_sequence_length.
  // We only support the encoder-self-attn subset today: no past KV, no mask,
  // no attention_bias.  Bail out (let it stay as onnx.Custom -> CPU fallback)
  // if anything else is present.
  if (op->getNumOperands() < 3)
    return rewriter.notifyMatchFailure(
        op, "Attention expects at least 3 operands (input, weights, bias)");

  auto isLive = [](mlir::Value v) {
    return v && !mlir::isa<mlir::NoneType>(v.getType());
  };

  for (size_t i = 3; i < op->getNumOperands(); ++i) {
    if (isLive(op->getOperand(i)))
      return rewriter.notifyMatchFailure(
          op, "unsupported optional Attention operand (mask / past / bias / "
              "past_seqlen) — only encoder-self-attn QKV-only is lowered");
  }

  mlir::Value hidden = op->getOperand(0);
  mlir::Value qkvW = op->getOperand(1);
  mlir::Value qkvB = op->getOperand(2);

  // === Attributes ===

  auto numHeadsAttrOnnx = op->getAttrOfType<mlir::IntegerAttr>("num_heads");
  if (!numHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing num_heads attribute");
  const int64_t numHeads = numHeadsAttrOnnx.getValue().getSExtValue();
  if (numHeads <= 0)
    return rewriter.notifyMatchFailure(op, "num_heads must be > 0");

  // Required for our path: do_rotary=0, unidirectional=0 (bidirectional),
  // qkv_hidden_sizes.size() == 3 with equal entries.  Reject anything else
  // — Whisper encoder always has equal sizes and we don't want to silently
  // mis-lower a different model.
  auto getI64 = [&](const char *name, int64_t defaultVal) {
    auto a = op->getAttrOfType<mlir::IntegerAttr>(name);
    return a ? a.getValue().getSExtValue() : defaultVal;
  };
  if (getI64("do_rotary", 0) != 0)
    return rewriter.notifyMatchFailure(op, "do_rotary != 0 not supported");
  if (getI64("unidirectional", 0) != 0)
    return rewriter.notifyMatchFailure(
        op,
        "unidirectional != 0 (causal Attention) not supported on this path");
  if (getI64("past_present_share_buffer", 0) != 0)
    return rewriter.notifyMatchFailure(
        op, "past_present_share_buffer != 0 not supported");

  auto qkvSizesAttr = op->getAttrOfType<mlir::ArrayAttr>("qkv_hidden_sizes");
  if (!qkvSizesAttr || qkvSizesAttr.size() != 3)
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
  const int64_t qSize = qkvSizes[0]; // == kSize == vSize

  // Scale: ONNX attr if present, else 1/sqrt(head_size).  We pass through to
  // hip.gqa (which already honors `scale=0.0` as the auto-compute sentinel).
  auto scaleAttrOnnx = op->getAttrOfType<mlir::FloatAttr>("scale");
  const float scale =
      scaleAttrOnnx ? scaleAttrOnnx.getValue().convertToFloat() : 0.0f;

  // === Shape sanity (we need static shapes for the constant seqlens_k) ===

  auto hiddenType = mlir::dyn_cast<mlir::RankedTensorType>(hidden.getType());
  if (!hiddenType || hiddenType.getRank() != 3 || !hiddenType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "hidden must be a static rank-3 tensor [B, S, H]");
  auto qkvWType = mlir::dyn_cast<mlir::RankedTensorType>(qkvW.getType());
  if (!qkvWType || qkvWType.getRank() != 2 || !qkvWType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "qkv_w must be a static rank-2 tensor [H, 3H]");

  const int64_t batch = hiddenType.getDimSize(0);
  const int64_t seqLen = hiddenType.getDimSize(1);
  const int64_t hiddenSize = hiddenType.getDimSize(2);
  const int64_t qkvHidden = qSize * 3;

  // ORT com.microsoft.Attention spec convention: qkv_w is [H, 3H], i.e.
  // [input_hidden, q_hidden+k_hidden+v_hidden].  This is what Whisper actually
  // exports (qkv_proj.weight = [1280, 3840]).  qkv_w.shape[0] must equal H and
  // qkv_w.shape[1] must equal qkvHidden (3H) — the matmul consumes it directly.
  if (qkvWType.getDimSize(0) != hiddenSize ||
      qkvWType.getDimSize(1) != qkvHidden)
    return rewriter.notifyMatchFailure(
        op, "qkv_w shape must be [hidden, sum(qkv_hidden_sizes)]");

  if (qSize % numHeads != 0)
    return rewriter.notifyMatchFailure(
        op, "qkv_hidden_sizes[0] must be divisible by num_heads");

  // === Context arg ===

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Type elemType = hiddenType.getElementType();

  // === 1. Fused QKV projection: [B, S, H] @ [H, 3H] = [B, S, 3H] ==========
  // The fused weight is already [H, 3H] (ORT Attention spec / Whisper export),
  // so we feed qkv_w straight into hip.matmul — NO transpose needed.
  auto projType =
      mlir::RankedTensorType::get({batch, seqLen, qkvHidden}, elemType);
  mlir::Value projInit = mlir::tensor::EmptyOp::create(
      rewriter, loc, projType.getShape(), projType.getElementType());
  mlir::Value qkvProj =
      mlir::hip::MatmulOp::create(rewriter, loc, projType, context, hidden,
                                  qkvW, projInit)
          .getResult(0);

  // === 2. Add bias (broadcasts on last dim): [B, S, 3H] + [3H] = [B,S,3H] ==
  mlir::Value addInit = mlir::tensor::EmptyOp::create(
      rewriter, loc, projType.getShape(), projType.getElementType());
  mlir::Value qkvBiased =
      mlir::hip::AddOp::create(rewriter, loc, projType, context, qkvProj, qkvB,
                               addInit)
          .getResult(0);

  // === 3. Split qkv_biased along the last axis into Q / K / V =============
  // Each slice has shape [B, S, qSize]; qSize == kSize == vSize per the
  // equality check above.
  auto sliceType =
      mlir::RankedTensorType::get({batch, seqLen, qSize}, elemType);
  auto buildSlice = [&](int64_t offsetOnLastAxis) -> mlir::Value {
    llvm::SmallVector<mlir::OpFoldResult, 3> offsets = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(offsetOnLastAxis)};
    llvm::SmallVector<mlir::OpFoldResult, 3> sizes = {
        rewriter.getIndexAttr(batch), rewriter.getIndexAttr(seqLen),
        rewriter.getIndexAttr(qSize)};
    llvm::SmallVector<mlir::OpFoldResult, 3> strides(3,
                                                     rewriter.getIndexAttr(1));
    mlir::OperationState sliceState(
        loc, mlir::tensor::ExtractSliceOp::getOperationName());
    mlir::tensor::ExtractSliceOp::build(rewriter, sliceState, sliceType,
                                        qkvBiased, offsets, sizes, strides);
    mlir::Operation *sliceOp = rewriter.create(sliceState);
    return sliceOp->getResult(0);
  };
  mlir::Value query = buildSlice(0);
  mlir::Value key = buildSlice(qSize);
  mlir::Value value = buildSlice(2 * qSize);

  // === 4. Build seqlens_k = [S, S, ..., S]  and total_seq_len = S =========
  //
  // WHY a compile-time constant (not a runtime input): encoder self-attention
  // has no padding mask — every token along the time dim S is a valid input
  // (Whisper pads its mel-spectrogram up-front, before the encoder), so the
  // valid-KV-length is exactly S for every batch element on every call.  No
  // per-call runtime value is needed; we emit arith.constant tensors that
  // const-fold downstream.
  //
  // hip.gqa derives the per-batch KV length from seqlens_k (1-D [B] i32)
  // and the cache-buffer total length from total_sequence_length (scalar i32).
  auto i32Ty = rewriter.getIntegerType(32);
  auto seqlensKType = mlir::RankedTensorType::get({batch}, i32Ty);
  llvm::SmallVector<llvm::APInt, 1> seqlensVals(
      static_cast<size_t>(batch), llvm::APInt(32, seqLen, /*isSigned=*/true));
  auto seqlensKAttr = mlir::DenseElementsAttr::get(
      seqlensKType, llvm::ArrayRef<llvm::APInt>(seqlensVals));
  mlir::Value seqlensK =
      mlir::arith::ConstantOp::create(rewriter, loc, seqlensKAttr);

  auto totalSeqLenType = mlir::RankedTensorType::get({}, i32Ty);
  auto totalSeqLenAttr = mlir::DenseElementsAttr::get(
      totalSeqLenType, llvm::APInt(32, seqLen, /*isSigned=*/true));
  mlir::Value totalSeqLen =
      mlir::arith::ConstantOp::create(rewriter, loc, totalSeqLenAttr);

  // === 5. Build hip.gqa ====================================================
  //
  // Output 1 is the only ONNX-visible result (encoder self-attn has no KV
  // cache exposed).  hip.gqa always emits present_key/present_value tensors
  // (no_causal does NOT change the op signature), so we create DPS init
  // buffers for them but discard their results.
  //
  // Cache buffer layout: [B, kv_num_heads, S, head_size] (BNSH).
  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (outputType.getShape() != hiddenType.getShape())
    return rewriter.notifyMatchFailure(
        op, "Attention output shape must match input hidden shape");

  auto presentType = mlir::RankedTensorType::get(
      {batch, numHeads, seqLen, qSize / numHeads}, elemType);

  // Delegate hip.gqa construction to the shared builder (see
  // OnnxToHipUtils.h).  Encoder self-attn has no KV cache, so we pass
  // pastKey/pastValue=nullptr and let the builder zero out those operand
  // segments.  The op has a single ONNX-visible result (output); the builder
  // positionally maps result[0]=output and drops the synthesized
  // present_key/present_value on the floor.
  return buildHipGqaCall(op, rewriter, context, query, key, value,
                         /*pastKey=*/nullptr, /*pastValue=*/nullptr, seqlensK,
                         totalSeqLen, numHeads, scale, /*noCausal=*/true,
                         outputType, presentType, presentType);
}

} // namespace

void populateAttentionConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<AttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
