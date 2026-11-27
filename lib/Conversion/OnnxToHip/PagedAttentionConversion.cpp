/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Custom(PagedAttention, com.microsoft) -> hip.paged_attention
struct PagedAttentionToHip : public mlir::RewritePattern {
  PagedAttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
PagedAttentionToHip::matchAndRewrite(mlir::Operation *op,
                                     mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "PagedAttention")
    return rewriter.notifyMatchFailure(op, "not a PagedAttention operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for PagedAttention");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  size_t numOps = op->getNumOperands();
  // Inputs 0-7 carry the base contract, 8-9 are the rotary tables, 10-15 are
  // optional features this lowering does not compute, and 16 is
  // attention_metadata.
  if (numOps < 8 || numOps > 17)
    return rewriter.notifyMatchFailure(
        op, "PagedAttention expects between 8 and 17 operands");

  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps)
      return nullptr;
    mlir::Value val = op->getOperand(idx);
    if (mlir::isa<mlir::NoneType>(val.getType()))
      return nullptr;
    return val;
  };

  for (size_t idx = 10; idx < numOps && idx < 16; ++idx) {
    if (getOptionalOperand(idx))
      return rewriter.notifyMatchFailure(
          op, "PagedAttention optional input is not supported");
  }
  for (size_t idx : {0, 3, 4, 5, 6, 7}) {
    if (!getOptionalOperand(idx))
      return rewriter.notifyMatchFailure(
          op, "PagedAttention required input is missing");
  }

  mlir::Value query = op->getOperand(0);
  mlir::Value key = getOptionalOperand(1);
  mlir::Value value = getOptionalOperand(2);
  mlir::Value keyCache = op->getOperand(3);
  mlir::Value valueCache = op->getOperand(4);
  mlir::Value cumSeq = op->getOperand(5);
  mlir::Value pastSeqlens = op->getOperand(6);
  mlir::Value blockTable = op->getOperand(7);
  mlir::Value cosCache = getOptionalOperand(8);
  mlir::Value sinCache = getOptionalOperand(9);
  // Kept as an operand so the graph input stays live. The kernel does not read
  // it.
  mlir::Value attentionMetadata = getOptionalOperand(16);

  if (op->getNumResults() != 3)
    return rewriter.notifyMatchFailure(
        op, "PagedAttention requires the key_cache_out and value_cache_out "
            "results");

  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto keyCacheOutType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(1).getType());
  auto valueCacheOutType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(2).getType());
  if (!outputType || !keyCacheOutType || !valueCacheOutType)
    return rewriter.notifyMatchFailure(
        op, "PagedAttention results must be ranked tensors");

  // The runtime ABI carries only the element size, so only f16 is accepted.
  for (mlir::Value tensor : {query, keyCache, valueCache}) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(tensor.getType());
    if (!tensorType || !tensorType.getElementType().isF16())
      return rewriter.notifyMatchFailure(
          op, "PagedAttention supports only f16 tensors");
  }

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

  auto getI64Attr = [&](const char *name,
                        int64_t defaultVal) -> mlir::IntegerAttr {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    return rewriter.getI64IntegerAttr(attr ? attr.getValue().getSExtValue()
                                           : defaultVal);
  };

  // Attributes the lowering does not implement are accepted only at their
  // ONNX defaults.
  auto intAttrIs = [&](const char *name, int64_t defaultVal) {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    return !attr || attr.getValue().getSExtValue() == defaultVal;
  };
  auto floatAttrIs = [&](const char *name, double defaultVal) {
    auto attr = op->getAttrOfType<mlir::FloatAttr>(name);
    return !attr || attr.getValueAsDouble() == defaultVal;
  };
  auto stringAttrIs = [&](const char *name, llvm::StringRef defaultVal) {
    auto attr = op->getAttrOfType<mlir::StringAttr>(name);
    // An empty string default reaches the compiler as a single NUL character.
    return !attr || attr.getValue().rtrim('\0') == defaultVal;
  };
  if (!floatAttrIs("softcap", 0.0) || !intAttrIs("is_causal", 1) ||
      !stringAttrIs("k_quant_type", "NONE") ||
      !stringAttrIs("v_quant_type", "NONE") ||
      !stringAttrIs("k_cache_dtype", "") ||
      !stringAttrIs("v_cache_dtype", "") ||
      !stringAttrIs("kv_cache_layout", "SEPARATE") ||
      !intAttrIs("v_head_size", 0))
    return rewriter.notifyMatchFailure(
        op, "PagedAttention variant is not supported");

  // scale == 0 selects 1/sqrt(head_size); lowering or the runtime resolves it.
  auto scaleOnnx = op->getAttrOfType<mlir::FloatAttr>("scale");
  auto scaleAttr = rewriter.getF32FloatAttr(
      scaleOnnx ? static_cast<float>(scaleOnnx.getValueAsDouble()) : 0.0f);

  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, query);
  // The present caches are fresh graph outputs. Sharing the past buffer is
  // decided by the output allocator, as for any other present tensor.
  mlir::Value keyCacheOutInit =
      createEmptyTensor(rewriter, loc, keyCacheOutType, keyCache);
  mlir::Value valueCacheOutInit =
      createEmptyTensor(rewriter, loc, valueCacheOutType, valueCache);

  mlir::SmallVector<mlir::Type> resultTypes = {outputType, keyCacheOutType,
                                               valueCacheOutType};

  mlir::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(query);
  if (key)
    operands.push_back(key);
  if (value)
    operands.push_back(value);
  operands.push_back(keyCache);
  operands.push_back(valueCache);
  operands.push_back(cumSeq);
  operands.push_back(pastSeqlens);
  operands.push_back(blockTable);
  if (cosCache)
    operands.push_back(cosCache);
  if (sinCache)
    operands.push_back(sinCache);
  if (attentionMetadata)
    operands.push_back(attentionMetadata);
  operands.push_back(outputInit);
  operands.push_back(keyCacheOutInit);
  operands.push_back(valueCacheOutInit);

  mlir::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("num_heads", numHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("kv_num_heads", kvNumHeadsAttr));
  attrs.push_back(
      rewriter.getNamedAttr("do_rotary", getI64Attr("do_rotary", 0)));
  attrs.push_back(rewriter.getNamedAttr("rotary_interleaved",
                                        getI64Attr("rotary_interleaved", 0)));
  attrs.push_back(rewriter.getNamedAttr("local_window_size",
                                        getI64Attr("local_window_size", -1)));
  attrs.push_back(rewriter.getNamedAttr("scale", scaleAttr));

  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(1); // ctx
  segmentSizes.push_back(1); // query
  segmentSizes.push_back(key ? 1 : 0);
  segmentSizes.push_back(value ? 1 : 0);
  segmentSizes.push_back(1); // key_cache
  segmentSizes.push_back(1); // value_cache
  segmentSizes.push_back(1); // cumulative_sequence_length
  segmentSizes.push_back(1); // past_seqlens
  segmentSizes.push_back(1); // block_table
  segmentSizes.push_back(cosCache ? 1 : 0);
  segmentSizes.push_back(sinCache ? 1 : 0);
  segmentSizes.push_back(attentionMetadata ? 1 : 0);
  segmentSizes.push_back(1); // output
  segmentSizes.push_back(1); // key_cache_out
  segmentSizes.push_back(1); // value_cache_out

  auto state = mlir::OperationState(loc, "hip.paged_attention");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));
  state.addTypes(resultTypes);

  auto hipOp = rewriter.create(state);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}
} // namespace

void populatePagedAttentionConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  patterns.add<PagedAttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
