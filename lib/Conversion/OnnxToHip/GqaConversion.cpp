/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Trace a scalar shape value (e.g. GQA's `total_sequence_length` operand)
/// back to a host-known input dimension, returning that `index` value — or
/// null if the chain does not resolve to a host dim (caller falls back to a
/// GPU readback).
///
/// Why this exists
/// ---------------
/// Every GQA exporter we ship (LLM + VLM) computes
///   total_sequence_length = Cast(Gather(Shape(attention_mask), k))
/// on the host. By the time this pattern fires, GatherShapeFold (which runs
/// to quiescence in the pre-lowering rounds, BEFORE the main conversion) has
/// already lowered that idiom to:
///   %d   = tensor.dim %attention_mask, %k : index
///   %d64 = arith.index_cast %d : index to i64
///   %fe  = tensor.from_elements %d64 : tensor<1xi64>
///   %tsl = <cast %fe to the i32 scalar GQA operand[6]>
/// Walking back through the value-preserving scalar ops lands on the
/// `tensor.dim`, whose `index` result already equals total_sequence_length
/// and dominates this op (it is in operand[6]'s producer chain). Reusing it
/// sizes the present-KV buffer with NO `hipStreamSynchronize` readback — and
/// sidesteps the host-scratch slot-reuse race that corrupts a readback of
/// total_sequence_length on the decode path.
///
/// Before (value `v` = GQA operand[6]):
///   %d = tensor.dim %am, %c1 ; ... ; %v = cast(from_elements(index_cast %d))
/// After (returned):
///   %d   (the index result of tensor.dim — reused as-is)
static mlir::Value traceScalarToHostIndex(mlir::Value v) {
  // The canonical chain is <= 5 ops; the cap defends against an unexpected
  // cycle or pathologically long chain.
  for (int step = 0; step < 16; ++step) {
    mlir::Operation *def = v.getDefiningOp();
    if (!def)
      return nullptr; // block argument: not a traceable host dim
    llvm::StringRef name = def->getName().getStringRef();

    // Found the host dim — its `index` result IS total_sequence_length.
    if (name == "tensor.dim")
      return def->getResult(0);

    // Value-preserving single-source scalar ops: peek through operand 0.
    // (the i64<->i32<->index casts, the from_elements wrapper, and the
    // onnx.Cast adapting Shape's i64 to GQA's i32 all preserve the scalar
    // value for sizing purposes.)
    if (name == "tensor.from_elements" || name == "tensor.extract" ||
        name == "arith.index_cast" || name == "arith.extsi" ||
        name == "arith.extui" || name == "arith.trunci" ||
        name == "onnx.Cast" || name == "hip.cast" || name == "onnx.Squeeze" ||
        name == "onnx.Unsqueeze") {
      if (def->getNumOperands() == 0)
        return nullptr;
      v = def->getOperand(0);
      continue;
    }
    return nullptr; // unrecognized producer: bail to readback
  }
  return nullptr;
}

/// onnx.Custom(GroupQueryAttention) -> hip.gqa
struct GroupQueryAttentionToHip : public mlir::RewritePattern {
  GroupQueryAttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult GroupQueryAttentionToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  // Check if this is GroupQueryAttention
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "GroupQueryAttention")
    return rewriter.notifyMatchFailure(op,
                                       "not a GroupQueryAttention operation");

  // Check domain is "com.microsoft"
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for GroupQueryAttention");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // Support variable operand count (7-14 inputs as per MS spec)
  // Minimum 7: query, key, value, past_key, past_value, seqlens_k,
  // total_seq_len Maximum 14: + cos_cache, sin_cache, position_ids,
  // attention_bias, head_sink, k_scale, v_scale
  size_t numOps = op->getNumOperands();
  if (numOps < 7 || numOps > 14)
    return rewriter.notifyMatchFailure(
        op, "GroupQueryAttention expects 7-14 operands");

  // Helper: get optional operand (check for NoneType)
  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps)
      return nullptr; // Operand not provided (trailing optionals omitted)

    mlir::Value val = op->getOperand(idx);

    // Check if it's ONNX NoneType (optional input marked as omitted)
    if (mlir::isa<mlir::NoneType>(val.getType()))
      return nullptr;

    return val; // Valid operand
  };

  // === Extract Inputs (MS GQA spec order 1-14) ===

  // Input 1: query (required)
  mlir::Value query = op->getOperand(0);

  // Inputs 2-3: key/value (optional - packed QKV)
  mlir::Value key = getOptionalOperand(1);
  mlir::Value value = getOptionalOperand(2);

  // Inputs 4-5: past_key/past_value (optional - first inference)
  mlir::Value pastKey = getOptionalOperand(3);
  mlir::Value pastValue = getOptionalOperand(4);

  // Inputs 6-7: seqlens_k, total_seq_len (required)
  if (numOps < 7)
    return rewriter.notifyMatchFailure(op, "missing seqlens_k/total_seq_len");
  mlir::Value seqlensK = op->getOperand(5);
  mlir::Value totalSeqLen = op->getOperand(6);

  // Inputs 8-10: cos_cache, sin_cache, position_ids (optional - RoPE)
  mlir::Value cosCache = getOptionalOperand(7);
  mlir::Value sinCache = getOptionalOperand(8);
  mlir::Value positionIds = getOptionalOperand(9);

  // Input 11: attention_bias (optional - ALiBi etc.)
  mlir::Value attentionBias = getOptionalOperand(10);

  // Input 12: head_sink (optional - smooth softmax)
  mlir::Value headSink = getOptionalOperand(11);

  // Inputs 13-14: k_scale, v_scale (optional - quantization)
  mlir::Value kScale = getOptionalOperand(12);
  mlir::Value vScale = getOptionalOperand(13);

  // === Extract Attributes ===

  // Required attributes - extract value and recreate as signless i64
  auto numHeadsAttrOnnx = op->getAttrOfType<mlir::IntegerAttr>("num_heads");
  if (!numHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing num_heads attribute");
  auto numHeadsAttr =
      rewriter.getI64IntegerAttr(numHeadsAttrOnnx.getValue().getSExtValue());

  auto kvNumHeadsAttrOnnx =
      op->getAttrOfType<mlir::IntegerAttr>("kv_num_heads");
  if (!kvNumHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing kv_num_heads attribute");
  auto kvNumHeadsAttr =
      rewriter.getI64IntegerAttr(kvNumHeadsAttrOnnx.getValue().getSExtValue());

  // Optional attributes (with default values)
  auto getFloatAttr = [&](const char *name,
                          float defaultVal) -> mlir::FloatAttr {
    auto attr = op->getAttrOfType<mlir::FloatAttr>(name);
    return attr ? attr : rewriter.getF32FloatAttr(defaultVal);
  };

  auto getI64Attr = [&](const char *name,
                        int64_t defaultVal) -> mlir::IntegerAttr {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    // Convert ONNX signed integer to signless integer for HIP dialect
    return attr ? rewriter.getI64IntegerAttr(attr.getValue().getSExtValue())
                : rewriter.getI64IntegerAttr(defaultVal);
  };

  auto getStrAttr = [&](const char *name,
                        const char *defaultVal) -> mlir::StringAttr {
    auto attr = op->getAttrOfType<mlir::StringAttr>(name);
    return attr ? attr : rewriter.getStringAttr(defaultVal);
  };

  // Calculate default scale = 1/sqrt(head_size) per ONNX spec
  // Query shape: [batch_size, seq_len, num_heads * head_size]
  // head_size = (num_heads * head_size) / num_heads = query_dim_2 / num_heads
  // Fallback 0.0 is the ORT sentinel meaning "auto-compute at runtime"
  // (gqa_attention_base.h: scale_ == 0.0f ? 1/sqrt(head_size) : scale_)
  auto queryType = mlir::cast<mlir::RankedTensorType>(query.getType());
  int64_t numHeads = numHeadsAttrOnnx.getValue().getSExtValue();
  float defaultScale = 0.0f;
  if (queryType.hasRank() && queryType.getRank() >= 3) {
    int64_t hiddenSize = queryType.getDimSize(2); // num_heads * head_size
    if (hiddenSize != mlir::ShapedType::kDynamic && numHeads > 0) {
      int64_t headSize = hiddenSize / numHeads;
      defaultScale = 1.0f / std::sqrt(static_cast<float>(headSize));
    }
  }

  auto scaleAttr = getFloatAttr("scale", defaultScale);
  auto doRotaryAttr = getI64Attr("do_rotary", 0);
  auto rotaryInterleavedAttr = getI64Attr("rotary_interleaved", 0);
  auto softcapAttr = getFloatAttr("softcap", 0.0f);
  auto localWindowSizeAttr = getI64Attr("local_window_size", -1);
  auto smoothSoftmaxAttr = getI64Attr("smooth_softmax", 0);
  auto qkOutputAttr = getI64Attr("qk_output", 0);
  auto kQuantTypeAttr = getStrAttr("k_quant_type", "NONE");
  auto vQuantTypeAttr = getStrAttr("v_quant_type", "NONE");
  auto kvCacheBitWidthAttr = getI64Attr("kv_cache_bit_width", 8);

  // === Check Outputs (3 or 4: output, present_key, present_value, [output_qk])
  // ===

  size_t numResults = op->getNumResults();
  if (numResults < 3 || numResults > 4)
    return rewriter.notifyMatchFailure(
        op, "GroupQueryAttention expects 3-4 results");

  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto presentKeyType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
  auto presentValueType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(2).getType());

  mlir::RankedTensorType outputQkType = nullptr;
  if (numResults == 4)
    outputQkType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(3).getType());

  // === Create DPS init tensors ===

  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, query);

  // present_key/value DPS-init shaping. The present KV-cache sequence extent is
  // total_sequence_length = past_sequence_length + new_sequence_length. The
  // plain createEmptyTensor donor (past_*) copies past_*'s seq extent into
  // present, which is only correct for past_present_share_buffer=true: there
  // past and present alias one buffer pre-sized to max_length, so
  // past_*.dim(seqAxis) == max_length >= total. For the separate-buffer case
  // past_*.dim(seqAxis) == past_sequence_length, so the present buffer comes
  // out sized to the *past* length and the newly appended tokens are dropped
  // (decode past=1 -> present=1 instead of 2; prefill past=0 -> present=0
  // instead of new_seq). Use max(past_*.dim(seqAxis), total_seq_len), which is
  // correct for BOTH runtime modes from one compiled DLL: shared-buffer ->
  // max == max_length (past wins); separate-buffer -> max == total (the scalar
  // total_seq_len operand wins). total_seq_len is GQA operand[6]; its host
  // value is obtained by tracing its producer chain back to a host-known input
  // dim (attention_mask.shape[1]) via traceScalarToHostIndex, falling back to
  // a synchronized GPU readback only if the trace fails (folds to a const when
  // static).
  //
  // Before (separate-buffer decode, past_sequence_length=1, new=1):
  //   %present = tensor.empty(%batch, tensor.dim %past_key[2] = 1)  // too
  //   small
  // After:
  //   %tsl = <traced attention_mask dim> ; %s = arith.maxsi(dim %past_key[2],
  //   %tsl) %present = tensor.empty(%batch, %s)                            //
  //   == total
  constexpr int64_t kPresentSeqAxis = 2; // BNSH: [batch, kv_heads, seq, head]

  mlir::Value totalSeqIdx; // index-typed total_seq_len, built lazily/once
  auto getTotalSeqIdx = [&]() -> mlir::Value {
    if (totalSeqIdx)
      return totalSeqIdx;
    // Primary: trace total_seq_len back to a host-known input dim (e.g.
    // attention_mask.shape[1]) and reuse that index directly — no GPU sync,
    // no host-scratch readback race.
    if (mlir::Value hostIdx = traceScalarToHostIndex(totalSeqLen)) {
      totalSeqIdx = hostIdx; // already index-typed and dominates this op
      return totalSeqIdx;
    }
    // Fallback: synchronized GPU readback of the scalar operand (used when
    // the exporter computes total_sequence_length in a form this trace does
    // not recognize).
    auto tslTy = mlir::cast<mlir::RankedTensorType>(totalSeqLen.getType());
    mlir::Value host =
        tslTy.getRank() == 0
            ? readbackScalarToHost(rewriter, loc, context, totalSeqLen)
            : readbackShapeEntryToHost(rewriter, loc, context, totalSeqLen, 0);
    totalSeqIdx = mlir::arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), host);
    return totalSeqIdx;
  };

  auto buildPresentInit = [&](mlir::RankedTensorType ptype,
                              mlir::Value pastVal) -> mlir::Value {
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t d : llvm::seq<int64_t>(ptype.getRank())) {
      if (!ptype.isDynamicDim(d))
        continue;
      if (d == kPresentSeqAxis) {
        mlir::Value pastSeq =
            mlir::tensor::DimOp::create(rewriter, loc, pastVal, d);
        dynSizes.push_back(mlir::arith::MaxSIOp::create(rewriter, loc, pastSeq,
                                                        getTotalSeqIdx()));
      } else {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, pastVal, d));
      }
    }
    return mlir::tensor::EmptyOp::create(rewriter, loc, ptype.getShape(),
                                         ptype.getElementType(), dynSizes);
  };

  mlir::Value presentKeyInit =
      pastKey
          ? buildPresentInit(presentKeyType, pastKey)
          : createEmptyTensor(rewriter, loc, presentKeyType, key ? key : query);
  mlir::Value presentValueInit =
      pastValue ? buildPresentInit(presentValueType, pastValue)
                : createEmptyTensor(rewriter, loc, presentValueType,
                                    value ? value : query);

  mlir::Value outputQkInit = nullptr;
  if (outputQkType)
    outputQkInit = createEmptyTensor(rewriter, loc, outputQkType, query);

  // === Create hip.gqa operation ===

  mlir::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(outputType);
  resultTypes.push_back(presentKeyType);
  resultTypes.push_back(presentValueType);
  if (outputQkType)
    resultTypes.push_back(outputQkType);

  // Build operands: context + inputs + outputs
  // Note: Only add non-null operands (optional ones may be nullptr)
  mlir::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(query);
  if (key)
    operands.push_back(key);
  if (value)
    operands.push_back(value);
  if (pastKey)
    operands.push_back(pastKey);
  if (pastValue)
    operands.push_back(pastValue);
  operands.push_back(seqlensK);
  operands.push_back(totalSeqLen);
  if (cosCache)
    operands.push_back(cosCache);
  if (sinCache)
    operands.push_back(sinCache);
  if (positionIds)
    operands.push_back(positionIds);
  if (attentionBias)
    operands.push_back(attentionBias);
  if (headSink)
    operands.push_back(headSink);
  if (kScale)
    operands.push_back(kScale);
  if (vScale)
    operands.push_back(vScale);
  operands.push_back(outputInit);
  operands.push_back(presentKeyInit);
  operands.push_back(presentValueInit);
  if (outputQkInit)
    operands.push_back(outputQkInit);

  // Build named attributes
  mlir::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("num_heads", numHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("kv_num_heads", kvNumHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("scale", scaleAttr));
  attrs.push_back(rewriter.getNamedAttr("do_rotary", doRotaryAttr));
  attrs.push_back(
      rewriter.getNamedAttr("rotary_interleaved", rotaryInterleavedAttr));
  attrs.push_back(rewriter.getNamedAttr("softcap", softcapAttr));
  attrs.push_back(
      rewriter.getNamedAttr("local_window_size", localWindowSizeAttr));
  attrs.push_back(rewriter.getNamedAttr("smooth_softmax", smoothSoftmaxAttr));
  attrs.push_back(rewriter.getNamedAttr("qk_output", qkOutputAttr));
  attrs.push_back(rewriter.getNamedAttr("k_quant_type", kQuantTypeAttr));
  attrs.push_back(rewriter.getNamedAttr("v_quant_type", vQuantTypeAttr));
  attrs.push_back(
      rewriter.getNamedAttr("kv_cache_bit_width", kvCacheBitWidthAttr));
  // Explicit no_causal=false: existing GroupQueryAttention ONNX op is always
  // causal. The Whisper bidirectional paths (encoder self-attn and decoder
  // cross-attn) construct hip.gqa via different conversions that set this to
  // true explicitly.
  attrs.push_back(
      rewriter.getNamedAttr("no_causal", rewriter.getBoolAttr(false)));

  // Create operation using builder
  // We need to compute the operand_segment_sizes attribute for
  // AttrSizedOperandSegments
  auto state = mlir::OperationState(loc, "hip.gqa");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes(resultTypes);

  // Add operand_segment_sizes for AttrSizedOperandSegments trait
  // Segments: [ctx(1), query(1), key(0|1), value(0|1), past_key(0|1),
  // past_value(0|1),
  //            seqlens_k(1), total_seq_len(1), cos_cache(0|1), sin_cache(0|1),
  //            position_ids(0|1), attention_bias(0|1), head_sink(0|1),
  //            k_scale(0|1), v_scale(0|1), output(1), present_key(1),
  //            present_value(1), output_qk(0|1)]
  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(1); // ctx
  segmentSizes.push_back(1); // query
  segmentSizes.push_back(key ? 1 : 0);
  segmentSizes.push_back(value ? 1 : 0);
  segmentSizes.push_back(pastKey ? 1 : 0);
  segmentSizes.push_back(pastValue ? 1 : 0);
  segmentSizes.push_back(1); // seqlens_k
  segmentSizes.push_back(1); // total_seq_len
  segmentSizes.push_back(cosCache ? 1 : 0);
  segmentSizes.push_back(sinCache ? 1 : 0);
  segmentSizes.push_back(positionIds ? 1 : 0);
  segmentSizes.push_back(attentionBias ? 1 : 0);
  segmentSizes.push_back(headSink ? 1 : 0);
  segmentSizes.push_back(kScale ? 1 : 0);
  segmentSizes.push_back(vScale ? 1 : 0);
  segmentSizes.push_back(1);                    // output
  segmentSizes.push_back(1);                    // present_key
  segmentSizes.push_back(1);                    // present_value
  segmentSizes.push_back(outputQkInit ? 1 : 0); // output_qk

  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));

  auto hipOp = rewriter.create(state);

  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateGqaConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<GroupQueryAttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
