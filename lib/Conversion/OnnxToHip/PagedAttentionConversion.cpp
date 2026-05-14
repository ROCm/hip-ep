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

mlir::LogicalResult PagedAttentionToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
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
  if (numOps < 8 || numOps > 10)
    return rewriter.notifyMatchFailure(
        op, "PagedAttention expects 8-10 operands");

  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps)
      return nullptr;
    mlir::Value val = op->getOperand(idx);
    if (mlir::isa<mlir::NoneType>(val.getType()))
      return nullptr;
    return val;
  };

  mlir::Value query = op->getOperand(0);
  mlir::Value key = getOptionalOperand(1);
  mlir::Value value = getOptionalOperand(2);
  mlir::Value keyCache = op->getOperand(3);
  mlir::Value valueCache = op->getOperand(4);
  mlir::Value cumSeq = op->getOperand(5);
  mlir::Value pastSeqlens = op->getOperand(6);
  mlir::Value blockTable = op->getOperand(7);
  mlir::Value cosCache = numOps > 8 ? getOptionalOperand(8) : nullptr;
  mlir::Value sinCache = numOps > 9 ? getOptionalOperand(9) : nullptr;

  size_t numResults = op->getNumResults();
  if (numResults < 1 || numResults > 3)
    return rewriter.notifyMatchFailure(
        op, "PagedAttention expects 1-3 results");

  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::RankedTensorType keyCacheOutType = nullptr;
  mlir::RankedTensorType valueCacheOutType = nullptr;
  if (numResults >= 2)
    keyCacheOutType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
  if (numResults >= 3)
    valueCacheOutType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(2).getType());

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

  auto getFloatAttr = [&](const char *name,
                          float defaultVal) -> mlir::FloatAttr {
    auto attr = op->getAttrOfType<mlir::FloatAttr>(name);
    return attr ? attr : rewriter.getF32FloatAttr(defaultVal);
  };

  auto getI64Attr = [&](const char *name,
                        int64_t defaultVal) -> mlir::IntegerAttr {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    return attr ? rewriter.getI64IntegerAttr(attr.getValue().getSExtValue())
                : rewriter.getI64IntegerAttr(defaultVal);
  };

  int64_t numHeads = numHeadsAttrOnnx.getValue().getSExtValue();
  float defaultScale = 0.0f;
  auto queryType = mlir::cast<mlir::RankedTensorType>(query.getType());
  if (queryType.hasRank() && queryType.getRank() >= 2) {
    int64_t d1 = queryType.getDimSize(1);
    if (d1 != mlir::ShapedType::kDynamic && numHeads > 0)
      defaultScale = 1.0f / std::sqrt(static_cast<float>(d1 / numHeads));
  }

  auto scaleAttr = getFloatAttr("scale", defaultScale);
  auto doRotaryAttr = getI64Attr("do_rotary", 0);
  auto rotaryInterleavedAttr = getI64Attr("rotary_interleaved", 0);
  auto localWindowSizeAttr = getI64Attr("local_window_size", -1);
  auto softcapAttr = getFloatAttr("softcap", 0.0f);

  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, query);
  mlir::Value keyCacheOutInit = nullptr;
  mlir::Value valueCacheOutInit = nullptr;
  if (keyCacheOutType)
    keyCacheOutInit =
        createEmptyTensor(rewriter, loc, keyCacheOutType, keyCache);
  if (valueCacheOutType)
    valueCacheOutInit =
        createEmptyTensor(rewriter, loc, valueCacheOutType, valueCache);

  mlir::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(outputType);
  if (keyCacheOutType)
    resultTypes.push_back(keyCacheOutType);
  if (valueCacheOutType)
    resultTypes.push_back(valueCacheOutType);

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
  operands.push_back(outputInit);
  if (keyCacheOutInit)
    operands.push_back(keyCacheOutInit);
  if (valueCacheOutInit)
    operands.push_back(valueCacheOutInit);

  mlir::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("num_heads", numHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("kv_num_heads", kvNumHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("do_rotary", doRotaryAttr));
  attrs.push_back(
      rewriter.getNamedAttr("rotary_interleaved", rotaryInterleavedAttr));
  attrs.push_back(
      rewriter.getNamedAttr("local_window_size", localWindowSizeAttr));
  attrs.push_back(rewriter.getNamedAttr("scale", scaleAttr));
  attrs.push_back(rewriter.getNamedAttr("softcap", softcapAttr));

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
  segmentSizes.push_back(1); // output
  segmentSizes.push_back(keyCacheOutInit ? 1 : 0);
  segmentSizes.push_back(valueCacheOutInit ? 1 : 0);

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

void mlir::hip::populatePagedAttentionConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<PagedAttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
