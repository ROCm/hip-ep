//===- LinearAttentionConversion.cpp - ONNX-to-HIP LinearAttention conversion -
//*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Custom(LinearAttention) -> hip.linear_attention
struct LinearAttentionToHip : public RewritePattern {
  LinearAttentionToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult
LinearAttentionToHip::matchAndRewrite(Operation* op,
                                      PatternRewriter& rewriter) const {
  auto funcNameAttr = op->getAttrOfType<StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "LinearAttention")
    return rewriter.notifyMatchFailure(op, "not a LinearAttention operation");

  auto domainAttr = op->getAttrOfType<StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for LinearAttention");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();

  // Support variable operand count (3-6 inputs as per spec)
  size_t numOps = op->getNumOperands();
  if (numOps < 3 || numOps > 6)
    return rewriter.notifyMatchFailure(op,
                                       "LinearAttention expects 3-6 operands");

  auto getOptionalOperand = [&](size_t idx) -> Value {
    if (idx >= numOps)
      return nullptr;
    Value val = op->getOperand(idx);
    if (isa<NoneType>(val.getType()))
      return nullptr;
    return val;
  };

  // === Extract Inputs (spec order 1-6) ===
  Value query = op->getOperand(0);
  Value key = op->getOperand(1);
  Value value = op->getOperand(2);
  Value pastState = getOptionalOperand(3);
  Value decay = getOptionalOperand(4);
  Value beta = getOptionalOperand(5);

  // === Extract Attributes ===
  auto qNumHeadsAttrOnnx = op->getAttrOfType<IntegerAttr>("q_num_heads");
  if (!qNumHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing q_num_heads attribute");
  auto qNumHeadsAttr =
      rewriter.getI64IntegerAttr(qNumHeadsAttrOnnx.getValue().getSExtValue());

  auto kvNumHeadsAttrOnnx = op->getAttrOfType<IntegerAttr>("kv_num_heads");
  if (!kvNumHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing kv_num_heads attribute");
  auto kvNumHeadsAttr =
      rewriter.getI64IntegerAttr(kvNumHeadsAttrOnnx.getValue().getSExtValue());

  auto getFloatAttr = [&](const char* name, float defaultVal) -> FloatAttr {
    auto attr = op->getAttrOfType<FloatAttr>(name);
    return attr ? attr : rewriter.getF32FloatAttr(defaultVal);
  };

  auto getI64Attr = [&](const char* name, int64_t defaultVal) -> IntegerAttr {
    auto attr = op->getAttrOfType<IntegerAttr>(name);
    return attr ? rewriter.getI64IntegerAttr(attr.getValue().getSExtValue())
                : rewriter.getI64IntegerAttr(defaultVal);
  };

  auto getStrAttr = [&](const char* name,
                        const char* defaultVal) -> StringAttr {
    auto attr = op->getAttrOfType<StringAttr>(name);
    return attr ? attr : rewriter.getStringAttr(defaultVal);
  };

  // scale = 0.0 means "auto-compute 1/sqrt(d_k) at runtime"
  auto queryType = cast<RankedTensorType>(query.getType());
  int64_t qNumHeads = qNumHeadsAttrOnnx.getValue().getSExtValue();
  float defaultScale = 0.0f;
  if (queryType.hasRank() && queryType.getRank() >= 3) {
    int64_t hiddenSize = queryType.getDimSize(2);
    if (hiddenSize != ShapedType::kDynamic && qNumHeads > 0) {
      int64_t headSize = hiddenSize / qNumHeads;
      defaultScale = 1.0f / std::sqrt(static_cast<float>(headSize));
    }
  }

  auto scaleAttr = getFloatAttr("scale", defaultScale);
  auto chunkSizeAttr = getI64Attr("chunk_size", 0);
  auto updateRuleAttr = getStrAttr("update_rule", "gated_delta");

  // === Check Outputs (always 2: output, present_state) ===
  size_t numResults = op->getNumResults();
  if (numResults != 2)
    return rewriter.notifyMatchFailure(
        op, "LinearAttention expects exactly 2 results");

  auto outputType = cast<RankedTensorType>(op->getResult(0).getType());
  auto presentStateType = cast<RankedTensorType>(op->getResult(1).getType());

  // === Create DPS init tensors ===
  Value outputInit = createEmptyTensor(rewriter, loc, outputType, query);
  Value presentStateInit = createEmptyTensor(rewriter, loc, presentStateType,
                                             pastState ? pastState : key);

  // === Create hip.linear_attention operation ===
  SmallVector<Type> resultTypes;
  resultTypes.push_back(outputType);
  resultTypes.push_back(presentStateType);

  SmallVector<Value> operands;
  operands.push_back(context);
  operands.push_back(query);
  operands.push_back(key);
  operands.push_back(value);
  if (pastState)
    operands.push_back(pastState);
  if (decay)
    operands.push_back(decay);
  if (beta)
    operands.push_back(beta);
  operands.push_back(outputInit);
  operands.push_back(presentStateInit);

  SmallVector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("q_num_heads", qNumHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("kv_num_heads", kvNumHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("scale", scaleAttr));
  attrs.push_back(rewriter.getNamedAttr("chunk_size", chunkSizeAttr));
  attrs.push_back(rewriter.getNamedAttr("update_rule", updateRuleAttr));

  auto state = OperationState(loc, "hip.linear_attention");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes(resultTypes);

  // Segments: [ctx(1), query(1), key(1), value(1), past_state(0|1),
  //            decay(0|1), beta(0|1), output(1), present_state(1)]
  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(1); // ctx
  segmentSizes.push_back(1); // query
  segmentSizes.push_back(1); // key
  segmentSizes.push_back(1); // value
  segmentSizes.push_back(pastState ? 1 : 0);
  segmentSizes.push_back(decay ? 1 : 0);
  segmentSizes.push_back(beta ? 1 : 0);
  segmentSizes.push_back(1); // output
  segmentSizes.push_back(1); // present_state

  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));

  auto hipOp = rewriter.create(state);
  rewriter.replaceOp(op, hipOp->getResults());
  return success();
}

} // namespace

void mlir::hip::populateLinearAttentionConversionPatterns(
    RewritePatternSet& patterns, MLIRContext* ctx) {
  patterns.add<LinearAttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
