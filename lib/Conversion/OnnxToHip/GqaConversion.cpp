/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

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

  // Calculate default scale = 1/sqrt(head_size) per ONNX spec.
  // Fallback 0.0 is the ORT sentinel meaning "auto-compute at runtime"
  // (gqa_attention_base.h: scale_ == 0.0f ? 1/sqrt(head_size) : scale_)
  //
  // head_size is recovered from query's last dim, but the divisor differs by
  // input layout:
  //   * separate Q/K/V: query dim2 = num_heads * head_size
  //       -> divide by num_heads
  //   * packed QKV:     query dim2 = (num_heads + 2*kv_num_heads) * head_size
  //       (key/value are absent) -> divide by (num_heads + 2*kv_num_heads)
  // Using num_heads for the packed layout computes head_size too large
  // (e.g. H=40,G=10,d=128: 7680/40 = 192 instead of 128) and bakes a wrong
  // 1/sqrt(head_size) into the op, silently mis-scaling every score -- packed
  // prefill then diverges from ORT-CPU (cosine ~0.99) even though Q/K/V are
  // byte-identical to the separate path. Divide by the packing factor so the
  // recovered head_size is correct for both layouts.
  auto queryType = mlir::cast<mlir::RankedTensorType>(query.getType());
  int64_t numHeads = numHeadsAttrOnnx.getValue().getSExtValue();
  int64_t kvNumHeads = kvNumHeadsAttrOnnx.getValue().getSExtValue();
  bool packedQkv = (key == nullptr && value == nullptr);
  int64_t headDivisor = packedQkv ? (numHeads + 2 * kvNumHeads) : numHeads;
  float defaultScale = 0.0f;
  if (queryType.hasRank() && queryType.getRank() >= 3) {
    int64_t hiddenSize = queryType.getDimSize(2);
    if (hiddenSize != mlir::ShapedType::kDynamic && headDivisor > 0) {
      int64_t headSize = hiddenSize / headDivisor;
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
  mlir::Value presentKeyInit = createEmptyTensor(
      rewriter, loc, presentKeyType, pastKey ? pastKey : (key ? key : query));
  mlir::Value presentValueInit =
      createEmptyTensor(rewriter, loc, presentValueType,
                        pastValue ? pastValue : (value ? value : query));

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
