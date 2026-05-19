/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include <cmath>

namespace mlir {
namespace hip {
namespace {

/// onnx.Custom(com.microsoft.MultiHeadAttention) -> hip.multi_head_attention
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

  // === Check Outputs (1-4: output, [present_key, present_value, qk]) ===

  size_t numResults = op->getNumResults();
  if (numResults < 1 || numResults > 4)
    return rewriter.notifyMatchFailure(
        op, "MultiHeadAttention expects 1-4 results");

  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  mlir::RankedTensorType presentKeyType = nullptr;
  mlir::RankedTensorType presentValueType = nullptr;
  mlir::RankedTensorType qkType = nullptr;
  if (numResults >= 2)
    presentKeyType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
  if (numResults >= 3)
    presentValueType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(2).getType());
  if (numResults >= 4)
    qkType = mlir::cast<mlir::RankedTensorType>(op->getResult(3).getType());

  // === Create DPS init tensors ===

  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, query);

  // For present_key/present_value, derive dynamic dims from the most
  // appropriate source: past_key/past_value when present (same buffer
  // shape), otherwise key/value, otherwise query (packed QKV).
  mlir::Value presentKeyInit = nullptr;
  mlir::Value presentValueInit = nullptr;
  mlir::Value qkInit = nullptr;
  if (presentKeyType)
    presentKeyInit = createEmptyTensor(rewriter, loc, presentKeyType,
                                       pastKey ? pastKey : (key ? key : query));
  if (presentValueType)
    presentValueInit =
        createEmptyTensor(rewriter, loc, presentValueType,
                          pastValue ? pastValue : (value ? value : query));
  if (qkType)
    qkInit = createEmptyTensor(rewriter, loc, qkType, query);

  // === Build the new hip.multi_head_attention op ===

  mlir::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(outputType);
  if (presentKeyType)
    resultTypes.push_back(presentKeyType);
  if (presentValueType)
    resultTypes.push_back(presentValueType);
  if (qkType)
    resultTypes.push_back(qkType);

  // Build operands: ctx + inputs (only non-null) + outputs (only non-null).
  mlir::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(query);
  if (key)
    operands.push_back(key);
  if (value)
    operands.push_back(value);
  if (bias)
    operands.push_back(bias);
  if (keyPaddingMask)
    operands.push_back(keyPaddingMask);
  if (attentionBias)
    operands.push_back(attentionBias);
  if (pastKey)
    operands.push_back(pastKey);
  if (pastValue)
    operands.push_back(pastValue);
  if (pastSequenceLength)
    operands.push_back(pastSequenceLength);
  if (cacheIndirection)
    operands.push_back(cacheIndirection);
  operands.push_back(outputInit);
  if (presentKeyInit)
    operands.push_back(presentKeyInit);
  if (presentValueInit)
    operands.push_back(presentValueInit);
  if (qkInit)
    operands.push_back(qkInit);

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
  segmentSizes.push_back(key ? 1 : 0);
  segmentSizes.push_back(value ? 1 : 0);
  segmentSizes.push_back(bias ? 1 : 0);
  segmentSizes.push_back(keyPaddingMask ? 1 : 0);
  segmentSizes.push_back(attentionBias ? 1 : 0);
  segmentSizes.push_back(pastKey ? 1 : 0);
  segmentSizes.push_back(pastValue ? 1 : 0);
  segmentSizes.push_back(pastSequenceLength ? 1 : 0);
  segmentSizes.push_back(cacheIndirection ? 1 : 0);
  segmentSizes.push_back(1); // output
  segmentSizes.push_back(presentKeyInit ? 1 : 0);
  segmentSizes.push_back(presentValueInit ? 1 : 0);
  segmentSizes.push_back(qkInit ? 1 : 0);

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
