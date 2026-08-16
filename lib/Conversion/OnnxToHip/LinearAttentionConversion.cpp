/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Custom(LinearAttention) -> hip.linear_attention
struct LinearAttentionToHip : public mlir::RewritePattern {
  LinearAttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
LinearAttentionToHip::matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "LinearAttention")
    return rewriter.notifyMatchFailure(op, "not a LinearAttention operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for LinearAttention");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // Support variable operand count (3-6 inputs as per spec)
  size_t numOps = op->getNumOperands();
  if (numOps < 3 || numOps > 6)
    return rewriter.notifyMatchFailure(op,
                                       "LinearAttention expects 3-6 operands");

  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps)
      return nullptr;
    mlir::Value val = op->getOperand(idx);
    if (mlir::isa<mlir::NoneType>(val.getType()))
      return nullptr;
    return val;
  };

  // === Extract Inputs (spec order 1-6) ===
  mlir::Value query = op->getOperand(0);
  mlir::Value key = op->getOperand(1);
  mlir::Value value = op->getOperand(2);
  mlir::Value pastState = getOptionalOperand(3);
  mlir::Value decay = getOptionalOperand(4);
  mlir::Value beta = getOptionalOperand(5);

  // === Extract Attributes ===
  auto qNumHeadsAttrOnnx = op->getAttrOfType<mlir::IntegerAttr>("q_num_heads");
  if (!qNumHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing q_num_heads attribute");
  auto qNumHeadsAttr =
      rewriter.getI64IntegerAttr(qNumHeadsAttrOnnx.getValue().getSExtValue());

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

  auto getStrAttr = [&](const char *name,
                        const char *defaultVal) -> mlir::StringAttr {
    auto attr = op->getAttrOfType<mlir::StringAttr>(name);
    return attr ? attr : rewriter.getStringAttr(defaultVal);
  };

  // scale = 0.0 means "auto-compute 1/sqrt(d_k) at runtime"
  auto queryType = mlir::cast<mlir::RankedTensorType>(query.getType());
  int64_t qNumHeads = qNumHeadsAttrOnnx.getValue().getSExtValue();
  float defaultScale = 0.0f;
  if (queryType.hasRank() && queryType.getRank() >= 3) {
    int64_t hiddenSize = queryType.getDimSize(2);
    if (hiddenSize != mlir::ShapedType::kDynamic && qNumHeads > 0) {
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

  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto presentStateType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());

  // === Create DPS init tensors ===
  int64_t kvNumHeads = kvNumHeadsAttr.getValue().getSExtValue();
  mlir::FailureOr<mlir::ReifiedRankedShapedTypeDims> outputShapes =
      reifyLinearAttentionOutputShapes(rewriter, loc, query, key, value,
                                       qNumHeads, kvNumHeads,
                                       [&]() { return op->emitError(); });
  if (mlir::failed(outputShapes))
    return rewriter.notifyMatchFailure(
        op, "cannot derive LinearAttention output shapes");

  mlir::FailureOr<mlir::Value> outputInit = createEmptyTensorFromReifiedShape(
      rewriter, loc, outputType, (*outputShapes)[0]);
  mlir::FailureOr<mlir::Value> presentStateInit =
      createEmptyTensorFromReifiedShape(rewriter, loc, presentStateType,
                                        (*outputShapes)[1]);
  if (mlir::failed(outputInit) || mlir::failed(presentStateInit))
    return rewriter.notifyMatchFailure(
        op, "LinearAttention result types disagree with inferred shapes");

  // === Create hip.linear_attention operation ===
  mlir::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(outputType);
  resultTypes.push_back(presentStateType);

  mlir::SmallVector<mlir::Value> operands;
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
  operands.push_back(*outputInit);
  operands.push_back(*presentStateInit);

  mlir::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("q_num_heads", qNumHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("kv_num_heads", kvNumHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("scale", scaleAttr));
  attrs.push_back(rewriter.getNamedAttr("chunk_size", chunkSizeAttr));
  attrs.push_back(rewriter.getNamedAttr("update_rule", updateRuleAttr));

  auto state = mlir::OperationState(loc, "hip.linear_attention");
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
  return mlir::success();
}

} // namespace

void populateLinearAttentionConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<LinearAttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
