/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// com.microsoft.CausalConvWithState -> hip.causal_conv_with_state
struct CausalConvWithStateToHip : public mlir::RewritePattern {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CausalConvWithStateToHip)
  CausalConvWithStateToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult CausalConvWithStateToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "CausalConvWithState")
    return rewriter.notifyMatchFailure(op,
                                       "not a CausalConvWithState operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for CausalConvWithState");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // CausalConvWithState has 2-4 inputs:
  //   input (required), weight (required), bias (optional), past_state
  //   (optional)
  size_t numOps = op->getNumOperands();
  if (numOps < 2 || numOps > 4)
    return rewriter.notifyMatchFailure(
        op, "CausalConvWithState expects 2-4 operands");

  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps)
      return nullptr;
    mlir::Value val = op->getOperand(idx);
    if (mlir::isa<mlir::NoneType>(val.getType()))
      return nullptr;
    return val;
  };

  mlir::Value input = op->getOperand(0);
  mlir::Value weight = op->getOperand(1);
  mlir::Value bias = getOptionalOperand(2);
  mlir::Value pastState = getOptionalOperand(3);

  // Extract attributes with defaults
  auto activationAttr = op->getAttrOfType<mlir::StringAttr>("activation");
  if (!activationAttr)
    activationAttr = rewriter.getStringAttr("none");

  auto ndimAttrOnnx = op->getAttrOfType<mlir::IntegerAttr>("ndim");
  auto ndimAttr =
      ndimAttrOnnx
          ? rewriter.getI64IntegerAttr(ndimAttrOnnx.getValue().getSExtValue())
          : rewriter.getI64IntegerAttr(1);

  // Outputs: output (same shape as input), present_state
  size_t numResults = op->getNumResults();
  if (numResults != 2)
    return rewriter.notifyMatchFailure(
        op, "CausalConvWithState expects exactly 2 results");

  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto presentStateType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());

  // Create DPS init tensors
  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, input);
  mlir::Value presentStateInit = createEmptyTensor(
      rewriter, loc, presentStateType, pastState ? pastState : input);

  // Build operands
  mlir::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(input);
  operands.push_back(weight);
  if (bias)
    operands.push_back(bias);
  if (pastState)
    operands.push_back(pastState);
  operands.push_back(outputInit);
  operands.push_back(presentStateInit);

  // Build attributes
  mlir::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("activation", activationAttr));
  attrs.push_back(rewriter.getNamedAttr("ndim", ndimAttr));

  // Build result types
  mlir::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(outputType);
  resultTypes.push_back(presentStateType);

  // Create operation with operand_segment_sizes
  auto state = mlir::OperationState(loc, "hip.causal_conv_with_state");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes(resultTypes);

  // Segments: [ctx(1), input(1), weight(1), bias(0|1), past_state(0|1),
  //            output(1), present_state(1)]
  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(1); // ctx
  segmentSizes.push_back(1); // input
  segmentSizes.push_back(1); // weight
  segmentSizes.push_back(bias ? 1 : 0);
  segmentSizes.push_back(pastState ? 1 : 0);
  segmentSizes.push_back(1); // output
  segmentSizes.push_back(1); // present_state

  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));

  auto hipOp = rewriter.create(state);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateCausalConvWithStateConversionPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx) {
  patterns.add<CausalConvWithStateToHip>(ctx);
}

} // namespace hip
} // namespace mlir
